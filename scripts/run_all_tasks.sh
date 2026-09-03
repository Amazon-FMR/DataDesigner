#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
recipe_dir="$repo_root/docs/assets/recipes/mcp_and_tooluse"
seed_path="/home/cyongzho/wikidata-seeds/search_agent_seeds_30k_part1.jsonl"
artifact_path="$repo_root/artifacts/search_agent_30k"
export_path="$artifact_path/search_agent_30k_part1.jsonl"
log_path="$artifact_path/run_part1.log"
seed_count=5000

curl --fail --silent --show-error http://127.0.0.1:3456/ready >/dev/null
mkdir -p "$artifact_path"

uv run "$recipe_dir/search_agent.py" \
  --seed-path "$seed_path" \
  --num-records "$seed_count" \
  --max-parallel-requests 2000 \
  --max-in-flight-tasks 2000 \
  --disable-early-shutdown \
  --artifact-path "$artifact_path" \
  --create \
  --dataset-name search_agent_30k \
  --export-path "$export_path" \
  2>&1 | tee "$log_path"
