#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
recipe_dir="$repo_root/docs/assets/recipes/mcp_and_tooluse"
seed_path="$recipe_dir/search_agent_seeds_100.jsonl"
artifact_path="$repo_root/artifacts/search_agent_100"
export_path="$artifact_path/search_agent_100.jsonl"
log_path="$artifact_path/run.log"
seed_count="$(wc -l < "$seed_path")"

curl --fail --silent --show-error http://127.0.0.1:3456/ready >/dev/null
mkdir -p "$artifact_path"

uv run "$recipe_dir/search_agent.py" \
  --seed-path "$seed_path" \
  --num-records "$seed_count" \
  --max-parallel-requests 8 \
  --artifact-path "$artifact_path" \
  --create \
  --dataset-name search_agent_100 \
  --export-path "$export_path" \
  2>&1 | tee "$log_path"
