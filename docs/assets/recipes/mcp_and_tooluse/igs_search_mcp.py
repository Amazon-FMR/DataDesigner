# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "boto3>=1.40,<2",
#     "mcp>=1.16,<2",
# ]
# ///
"""Local MCP server exposing Amazon IGS web search."""

from __future__ import annotations

import asyncio
import json
import os
import threading
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from enum import Enum, auto
from functools import partial
from typing import Any

import boto3
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.credentials import Credentials
from mcp.server.fastmcp import FastMCP

IGS_REGION = "us-east-1"
IGS_ROLE_ARN = "arn:aws:iam::730335607287:role/ApiGatewayAccess-Experimental-hdliu"
IGS_ROLE_SESSION_NAME = "data-designer-igs-search"
IGS_SEARCH_URL = "https://apigateway.prod.us-east-1.infogroundingservice.agi.amazon.dev/internal-search"
IGS_REQUEST_TIMEOUT_SEC = 120
IGS_OBSERVATION_LIMIT = 20_000
IGS_HIT_TEXT_LIMIT = 3_000
IGS_MAX_RETURNED_HITS = 10
IGS_MAX_CONCURRENT_REQUESTS = 100
_EXPIRED_TOKEN_ERROR_TYPES = frozenset({"ExpiredToken", "ExpiredTokenException"})
_credential_lock = threading.Lock()
_credentials: Any = None
_search_executor = ThreadPoolExecutor(
    max_workers=int(os.environ.get("IGS_MAX_CONCURRENT_REQUESTS", IGS_MAX_CONCURRENT_REQUESTS)),
    thread_name_prefix="igs-search",
)


class PostStatus(Enum):
    """Outcome of an IGS HTTP request."""

    OK = auto()
    REFRESH_CREDENTIALS = auto()
    ERROR = auto()


async def igs_search(query: str, query_locale: str = "en-US", accept_language: str = "en-US") -> str:
    """Search the web with IGS and return normalized, bounded results."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _search_executor,
        partial(_igs_search_sync, query, query_locale=query_locale, accept_language=accept_language),
    )


def _igs_search_sync(query: str, query_locale: str = "en-US", accept_language: str = "en-US") -> str:
    """Execute one blocking IGS request outside the MCP event loop."""
    query = query.strip()
    if not query:
        raise ValueError("query is required")
    if len(query) > 200:
        raise ValueError("query must be at most 200 characters")

    payload = {
        "traceId": str(uuid.uuid4()),
        "queryList": [{"query": query, "queryLocale": query_locale}],
        "metadata": {
            "acceptLanguage": accept_language,
            "profile": "NGSVirtualAssistants",
            "isTestTraffic": True,
        },
    }
    output, status = _post_signed(payload)
    if status is PostStatus.REFRESH_CREDENTIALS:
        _refresh_credentials()
        output, status = _post_signed(payload)
    if status is not PostStatus.OK:
        raise RuntimeError(_truncate(output))

    try:
        response = json.loads(output)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"IGS returned invalid JSON: {exc}") from exc
    return _render_observation(response)


def _post_signed(payload: dict[str, Any]) -> tuple[str, PostStatus]:
    """POST a SigV4-signed request to IGS."""
    body = json.dumps(payload)
    try:
        credentials = _get_credentials()
    except Exception as exc:
        return f"Failed to authenticate IGS request: {exc}", PostStatus.ERROR

    request = AWSRequest(
        method="POST",
        url=IGS_SEARCH_URL,
        data=body,
        headers={"Content-Type": "application/json"},
    )
    SigV4Auth(credentials, "execute-api", IGS_REGION).add_auth(request)
    http_request = urllib.request.Request(
        url=IGS_SEARCH_URL,
        data=body.encode(),
        headers=dict(request.headers),
        method="POST",
    )
    try:
        with urllib.request.urlopen(http_request, timeout=IGS_REQUEST_TIMEOUT_SEC) as response:
            return response.read().decode(), PostStatus.OK
    except urllib.error.HTTPError as exc:
        output = exc.read().decode(errors="replace")
        error_type = exc.headers.get("x-amzn-errortype", "").partition(":")[0]
        if error_type in _EXPIRED_TOKEN_ERROR_TYPES:
            return output, PostStatus.REFRESH_CREDENTIALS
        return f"IGS request failed with HTTP status {exc.code}:\n{output}", PostStatus.ERROR
    except urllib.error.URLError as exc:
        return f"IGS request failed: {exc.reason}", PostStatus.ERROR
    except TimeoutError:
        return f"IGS request timed out after {IGS_REQUEST_TIMEOUT_SEC}s", PostStatus.ERROR


def _get_credentials() -> Any:
    """Return cached credentials for the IGS API role."""
    global _credentials

    with _credential_lock:
        if _credentials is None:
            _credentials = _assume_role_credentials()
        return _credentials


def _refresh_credentials() -> None:
    """Refresh cached IGS role credentials."""
    global _credentials

    with _credential_lock:
        _credentials = _assume_role_credentials()


def _assume_role_credentials() -> Any:
    role_arn = os.environ.get("IGS_ROLE_ARN", IGS_ROLE_ARN)
    response = boto3.client("sts", region_name=IGS_REGION).assume_role(
        RoleArn=role_arn,
        RoleSessionName=IGS_ROLE_SESSION_NAME,
    )
    credentials = response["Credentials"]
    return Credentials(
        access_key=credentials["AccessKeyId"],
        secret_key=credentials["SecretAccessKey"],
        token=credentials["SessionToken"],
    ).get_frozen_credentials()


def _render_observation(response: Any) -> str:
    hits: list[dict[str, Any]] = []
    results = response.get("results", {}) if isinstance(response, dict) else {}
    if isinstance(results, dict):
        for observations in results.values():
            if not isinstance(observations, list):
                continue
            for observation in observations:
                if not isinstance(observation, dict):
                    continue
                for hit in observation.get("hits") or ():
                    normalized = _normalize_hit(hit)
                    if normalized is not None:
                        hits.append(normalized)

    if not hits:
        return "No results returned for this query."
    rendered = "\n\n".join(_render_hit(index, hit) for index, hit in enumerate(hits[:IGS_MAX_RETURNED_HITS], 1))
    return _truncate(rendered)


def _normalize_hit(hit: Any) -> dict[str, Any] | None:
    if not isinstance(hit, dict):
        return None
    text = hit.get("text")
    if not isinstance(text, str) or not text.strip():
        return None
    metadata = hit.get("hitMetadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    sources = metadata.get("sources")
    source = sources[0] if isinstance(sources, list) and sources and isinstance(sources[0], dict) else {}
    return {
        "title": _optional_string(metadata.get("title")),
        "url": _optional_string(metadata.get("sourceUrl") or metadata.get("contentUrl")),
        "score": metadata.get("score"),
        "published_date": _published_date(metadata.get("publishDate")),
        "source_name": _optional_string(source.get("sourceName")),
        "text": text.strip()[:IGS_HIT_TEXT_LIMIT],
    }


def _optional_string(value: Any) -> str | None:
    return value.strip() or None if isinstance(value, str) else None


def _published_date(value: Any) -> str | None:
    try:
        return datetime.fromtimestamp(int(value) / 1000, tz=timezone.utc).isoformat()
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def _render_hit(index: int, hit: dict[str, Any]) -> str:
    return "\n".join(
        (
            f"[{index}]",
            f"title: {hit['title']}",
            f"url: {hit['url']}",
            f"score: {hit['score']}",
            f"published_date: {hit['published_date']}",
            f"source_name: {hit['source_name']}",
            "text:",
            hit["text"],
        )
    )


def _truncate(output: str) -> str:
    if len(output) <= IGS_OBSERVATION_LIMIT:
        return output
    elided = len(output) - IGS_OBSERVATION_LIMIT
    return f"[WARNING: Output truncated. {elided} chars elided.]\n{output[:IGS_OBSERVATION_LIMIT]}"


def main() -> None:
    """Run the IGS tool over stdio."""
    server = FastMCP("igs-search", log_level="WARNING")
    server.tool()(igs_search)
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
