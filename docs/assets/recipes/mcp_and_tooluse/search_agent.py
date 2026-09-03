# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "data-designer",
# ]
# ///
"""Nemotron Super Search Agent Recipe: Trajectories with IGS Web Search

Generate multi-turn search agent trajectories where an LLM iteratively
searches the web, reads results, reasons about evidence, and synthesizes
answers -- the kind of data needed to train BrowseComp-style search agents.

This recipe implements the pipeline used to produce ~7,000 high-quality
tool-use trajectories for Nemotron Super post-training, starting from
50,000 Wikidata knowledge graph seeds.

Pipeline architecture:

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                   STAGE 1: SEED DATA (Wikidata KG Walks)                │
    │                                                                         │
    │  Random walks on the Wikidata knowledge graph produce multi-hop paths.  │
    │  Each seed has: seed_entity, final_answer_entity, readable_path,        │
    │  num_hops_in_graph, ground_truth.                                       │
    │  Built-in demo seeds included; bring your own for production.           │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                   STAGE 2: SEARCH RIDDLE GENERATION (LLM)               │
    │                                                                         │
    │  user_query_draft ────────► user_query_obfuscated                       │
    │  (chain clues from path,     (BrowseComp-style rewrite:                 │
    │   hide intermediate nodes,    concise, natural, no breadcrumbs,         │
    │   don't name the answer)      1-2 sentences max)                        │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                   STAGE 3: SEARCH TRAJECTORY ROLLOUTS (LLM + MCP)       │
    │                                                                         │
    │  Thought-Action-Observation loop with live IGS web search.              │
    │  ├─ igs_search tool via a local stdio MCP server                        │
    │  ├─ Maximum 25 tool call turns; 300s timeout                            │
    │  ├─ Full trace captured via with_trace=ALL_MESSAGES                     │
    │  └─ Structured JSON output: final_answer, supporting_urls,              │
    │     short_justification                                                 │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                   STAGE 4: STRUCTURED FORMATTING (LLM)                  │
    │                                                                         │
    │  Normalize raw agent output into clean JSON via LLMStructuredColumn.    │
    │  Handles markdown fences, trailing text, single-quoted dicts.           │
    │                                                                         │
    │  The agent_solution_raw__trace column remains in the DataDesigner       │
    │  artifact for debugging, but is omitted from the final JSONL export.    │
    └─────────────────────────────────────────────────────────────────────────┘

Prerequisites:
    - AWS credentials that can assume the configured IGS role.
    - A local Mantle proxy for Bedrock authentication (default: http://127.0.0.1:3456).

Run:
    # Basic usage with built-in demo seeds (generates 2 trajectories)
    uv run search_agent.py

    # Use a custom seed parquet
    uv run search_agent.py --seed-path /path/to/seeds.parquet --num-records 10

    # For help message and available options
    uv run search_agent.py --help
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from pydantic import BaseModel, Field

import data_designer.config as dd
from data_designer.interface import DataDesigner, DatasetCreationResults

# =============================================================================
# Structured Output Schema
# =============================================================================


class AgentSolution(BaseModel):
    """Structured output for the search agent's final answer."""

    final_answer: str = Field(..., min_length=1, description="The final answer entity.")
    supporting_urls: list[str] = Field(
        default_factory=list, description="Authoritative URLs used to verify the answer."
    )
    short_justification: str = Field(..., min_length=1, description="Brief explanation of reasoning (1-2 sentences).")


def export_final_jsonl(results: DatasetCreationResults, path: str | Path) -> Path:
    """Export the final dataset without the internal agent rollout trace."""
    output_path = Path(path)
    dataset = results.load_dataset().drop(columns=["agent_solution_raw__trace"], errors="ignore")
    dataset.to_json(output_path, orient="records", lines=True, force_ascii=False, date_format="iso")
    return output_path


# =============================================================================
# Prompt Templates
# =============================================================================

QUERY_DRAFT_PROMPT = """\
You are an expert Search Evaluator designing Grandmaster-Level search tests.
Create a complex, multi-step search riddle based on this knowledge path:

{{ readable_path }}

Start Entity: {{ seed_entity }}
Final Answer Entity: {{ final_answer_entity }}

CRITICAL RULES:
1. DO NOT name the intermediate nodes. Hide them behind descriptions.
2. DO NOT name the Final Answer.
3. Chain the clues logically -- describe each step relative to the previous one.
4. Audit the logic: if a step is weak or nonsensical, IGNORE IT.
5. Salvage and simplify: use only the strongest, most logical hops.
6. No hallucinations: do not invent relationships not in the path.
7. Aim for 4-8 meaningful hops.

VALIDATION - Output "INVALID_PATH" if:
- Final answer is generic/abstract (e.g. "technology", "people", "field")
- Path has weak/illogical relationships
- No coherent question can be formed

Return ONLY the question string (or "INVALID_PATH").\
"""

OBFUSCATE_PROMPT = """\
Rewrite this search riddle to better match BrowseComp-style tasks.

Original Riddle: {{ user_query_draft }}

Secret Path (do not leak entities): {{ readable_path }}
Start Entity: {{ seed_entity }}
Final Answer (do not leak): {{ final_answer_entity }}

HARD REQUIREMENTS:
1. NEVER reveal the step-by-step plan. No breadcrumb chains.
   Avoid: "X is member of Y; Y is based in Z; Z is the capital of..."
   Avoid meta language: "then search...", "next find...", "follow the chain..."
2. NEVER mention the final answer or any intermediate entity by name.
3. Keep it concise and natural: 1-2 sentences max (3 for very complex paths).
4. Use descriptive clues that require reasoning.
5. Include at least one disambiguating filter (date, count, or specific attribute).
6. If original == "INVALID_PATH", output exactly "INVALID_PATH".

Return ONLY the rewritten question string (or "INVALID_PATH").\
"""

AGENT_SYSTEM_PROMPT = """\
You are an expert search agent that uses web search to answer questions accurately.

You MUST output ONLY valid JSON matching this exact schema:

{
  "final_answer": "string - the specific answer entity",
  "supporting_urls": ["url1", "url2"],
  "short_justification": "string - brief 1-2 sentence explanation"
}

AVAILABLE TOOLS:
You have access to ONE tool called "igs_search" with parameter: query (string, required).

TOOL USAGE RULES:
1. Exact Tool Name: Always use "igs_search" (no suffixes or prefixes).
2. Exact Args: Only send {"query": "..."} for the tool call.
3. Maximum 25 tool calls. Budget your searches wisely.
4. Search Strategy:
   - Start with broad queries to understand the domain
   - Refine to specific entities/relationships
   - Cross-verify facts across multiple sources
   - Use different query formulations for the same information
5. No Reasoning Tags: Do NOT use <think> tags or XML formatting.
6. No Intermediate Text: Do NOT output explanatory text between tool calls.
7. Final Output: After completing your searches, output ONLY the JSON object.

EXECUTION FLOW:
1. Read the user's question
2. Make tool calls using "igs_search" to gather information
3. Verify information across multiple sources
4. Once confident, output the JSON result (no additional text)\
"""

FORMATTER_PROMPT = """\
You are a JSON normalizer.

You will be given a messy model output that should contain a JSON object with:
- final_answer (string)
- supporting_urls (list of strings)
- short_justification (string)

Rules:
- Return ONLY a JSON object. No markdown. No extra text.
- If the input contains code fences, tool chatter, or extra prose, ignore it.
- If the input contains invalid JSON, repair it.
- supporting_urls must be a list of valid http(s) URLs (dedupe, keep best 1-5).

Input:
{{ agent_solution_raw }}\
"""


# =============================================================================
# Data Designer Configuration
# =============================================================================


def build_config(
    model_alias: str,
) -> tuple[dd.DataDesignerConfigBuilder, dd.LocalStdioMCPProvider]:
    """Build the Data Designer configuration for search agent trajectory generation.

    Returns:
        A tuple of (config_builder, mcp_provider).
    """
    server_path = Path(__file__).with_name("igs_search_mcp.py").resolve()
    mcp_provider = dd.LocalStdioMCPProvider(
        name="igs",
        command="uv",
        args=["run", str(server_path)],
    )

    tool_config = dd.ToolConfig(
        tool_alias="igs-search",
        providers=["igs"],
        allow_tools=["igs_search"],
        max_tool_call_turns=25,
        timeout_sec=300.0,
    )

    config_builder = dd.DataDesignerConfigBuilder(tool_configs=[tool_config])

    # Stage 2: Draft question from knowledge path
    config_builder.add_column(
        dd.LLMTextColumnConfig(
            name="user_query_draft",
            model_alias=model_alias,
            prompt=QUERY_DRAFT_PROMPT,
        )
    )

    # Stage 2: BrowseComp-style obfuscation
    config_builder.add_column(
        dd.LLMTextColumnConfig(
            name="user_query_obfuscated",
            model_alias=model_alias,
            prompt=OBFUSCATE_PROMPT,
        )
    )

    # Stage 3: Agent trajectory with MCP tool calling
    config_builder.add_column(
        dd.LLMTextColumnConfig(
            name="agent_solution_raw",
            model_alias=model_alias,
            system_prompt=AGENT_SYSTEM_PROMPT,
            prompt="Problem: {{ user_query_obfuscated }}",
            tool_alias="igs-search",
            with_trace=dd.TraceType.ALL_MESSAGES,
        )
    )

    # Stage 4: Structured JSON formatting
    config_builder.add_column(
        dd.LLMStructuredColumnConfig(
            name="agent_solution",
            model_alias=model_alias,
            prompt=FORMATTER_PROMPT,
            output_format=AgentSolution,
        )
    )

    return config_builder, mcp_provider


# =============================================================================
# Demo Seed Data
# =============================================================================

DEMO_SEEDS = [
    {
        "seed_entity": "NVIDIA",
        "final_answer_entity": "Thomas Hart Benton",
        "readable_path": (
            "START ENTITY: NVIDIA (Q182477)\n"
            "  \u2b07 [chief executive officer (P169)]\n"
            "  NODE: Jensen Huang (Q332838)\n"
            "  \u2b07 [educated at (P69)]\n"
            "  NODE: Oregon State University (Q861888)\n"
            "  \u2b07 [located in the administrative territorial entity (P131)]\n"
            "  NODE: Benton County (Q115372)\n"
            "  \u2b07 [named after (P138)]\n"
            "  NODE: Thomas Hart Benton (Q178712)"
        ),
        "num_hops_in_graph": 4,
        "ground_truth": "Thomas Hart Benton",
    },
    {
        "seed_entity": "Python",
        "final_answer_entity": "Centrum Wiskunde & Informatica",
        "readable_path": (
            "START ENTITY: Python (Q28865)\n"
            "  \u2b07 [developer (P178)]\n"
            "  NODE: Guido van Rossum (Q19845)\n"
            "  \u2b07 [employer (P108)]\n"
            "  NODE: Centrum Wiskunde & Informatica (Q1060645)"
        ),
        "num_hops_in_graph": 2,
        "ground_truth": "Centrum Wiskunde & Informatica",
    },
    {
        "seed_entity": "toothache",
        "final_answer_entity": "ibuprofen",
        "readable_path": (
            "START ENTITY: toothache (Q143925)\n"
            "  \u2b07 [risk factor (P564)]\n"
            "  NODE: smoking (Q662860)\n"
            "  \u2b07 [has effect (P1542)]\n"
            "  NODE: Crohn's disease (Q1472)\n"
            "  \u2b07 [drug or therapy used for treatment (P2176)]\n"
            "  NODE: TNF inhibitor (Q1536078)\n"
            "  \u2b07 [is possible treatment of (P2175)]\n"
            "  NODE: Beh\u00e7et's disease (Q911427)\n"
            "  \u2b07 [symptoms and signs (P780)]\n"
            "  NODE: inflammation (Q101991)\n"
            "  \u2b07 [drug or therapy used for treatment (P2176)]\n"
            "  NODE: flurbiprofen (Q419890)\n"
            "  \u2b07 [significant drug interaction (P769)]\n"
            "  NODE: parecoxib (Q347941)\n"
            "  \u2b07 [significant drug interaction (P769)]\n"
            "  NODE: ibuprofen (Q186969)"
        ),
        "num_hops_in_graph": 8,
        "ground_truth": "ibuprofen",
    },
]


def write_demo_seeds(output_dir: Path) -> Path:
    """Write demo seed data to a JSONL file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_path = output_dir / "demo_seeds.jsonl"
    with open(seed_path, "w", encoding="utf-8") as f:
        for seed in DEMO_SEEDS:
            f.write(json.dumps(seed, ensure_ascii=False) + "\n")
    return seed_path


# =============================================================================
# Main Entry Point
# =============================================================================


def parse_args():
    """Parse command line arguments."""
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Generate search agent trajectories using IGS web search via MCP.")
    parser.add_argument("--model-alias", type=str, default="bedrock-opus", help="Model alias to use for generation")
    parser.add_argument(
        "--bedrock-proxy-url",
        default=os.environ.get("BEDROCK_PROXY_URL", "http://127.0.0.1:3456"),
        help="Local Mantle proxy base URL",
    )
    parser.add_argument("--num-records", type=int, default=2, help="Number of trajectories to generate")
    parser.add_argument(
        "--max-parallel-requests",
        type=int,
        default=2,
        help="Maximum concurrent requests to the configured model",
    )
    parser.add_argument("--seed-path", type=str, default=None, help="Path to seed parquet or JSONL file")
    parser.add_argument("--artifact-path", type=str, default=None, help="Path to save artifacts")
    parser.add_argument("--create", action="store_true", help="Persist a complete dataset instead of running a preview")
    parser.add_argument("--dataset-name", type=str, default="search_agent", help="Dataset name used in create mode")
    parser.add_argument("--export-path", type=str, default=None, help="Optional JSONL export path for create mode")
    return parser.parse_args()


def main() -> None:
    """Main entry point for the demo."""
    args = parse_args()
    if args.seed_path:
        seed_path = args.seed_path
    else:
        demo_dir = Path(tempfile.mkdtemp(prefix="search_agent_demo_"))
        seed_path = str(write_demo_seeds(demo_dir))
        print(f"Using demo seeds in: {demo_dir}")

    config_builder, mcp_provider = build_config(model_alias=args.model_alias)
    config_builder.add_model_config(
        dd.ModelConfig(
            alias="bedrock-opus",
            model="anthropic.claude-opus-4-8",
            provider="bedrock-mantle",
            inference_parameters=dd.ChatCompletionInferenceParams(
                max_tokens=4_096,
                max_parallel_requests=args.max_parallel_requests,
                timeout=300,
            ),
        )
    )
    config_builder.with_seed_dataset(
        dd.LocalFileSeedSource(path=seed_path),
        sampling_strategy=dd.SamplingStrategy.SHUFFLE,
    )

    model_provider = dd.ModelProvider(
        name="bedrock-mantle",
        endpoint=f"{args.bedrock_proxy_url.rstrip('/')}/anthropic/v1",
        provider_type="anthropic",
    )
    data_designer = DataDesigner(
        artifact_path=args.artifact_path,
        model_providers=[model_provider],
        mcp_providers=[mcp_provider],
    )
    if args.create:
        results = data_designer.create(
            config_builder,
            num_records=args.num_records,
            dataset_name=args.dataset_name,
        )
        print(f"Persisted {results.count_records()} records under: {results.artifact_storage.base_dataset_path}")
        if args.export_path:
            export_path = export_final_jsonl(results, args.export_path)
            print(f"Exported dataset to: {export_path}")
    else:
        results = data_designer.preview(config_builder, num_records=args.num_records)

    print("\n" + "=" * 60)
    print("GENERATED SEARCH AGENT TRAJECTORIES")
    print("=" * 60)
    results.display_sample_record()


if __name__ == "__main__":
    main()
