#!/usr/bin/env bash
set -euo pipefail

OWNER="${1:-raybeecham}"
REPOSITORY="${2:-quantum-oncology-benchmark}"

if ! command -v gh >/dev/null 2>&1; then
  echo "GitHub CLI is required. Install it, then run: gh auth login" >&2
  exit 1
fi

gh auth status

if [[ ! -d .git ]]; then
  git init -b main
  git add .
  git commit -m "Initialize quantum oncology benchmark"
fi

if ! git remote get-url origin >/dev/null 2>&1; then
  gh repo create "$OWNER/$REPOSITORY" \
    --public \
    --source . \
    --remote origin \
    --push \
    --description "Reproducible classical and quantum machine-learning benchmarks for cancer classification research."
else
  git push -u origin main
fi
