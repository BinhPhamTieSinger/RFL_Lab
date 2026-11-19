#!/usr/bin/env bash
# Mac side of the branch workflow (docs/OPERATIONS.md, section 4): bring in the runners' results.
#
#   bash scripts/hub_collect.sh        # fetch, merge origin/acer and origin/pc into main, list the new runs
#
# Then analyse, commit what you changed, and `git push origin main` (Netlify deploys main). The runners take the
# new main with `bash scripts/runner_sync.sh <branch> --update` between sweeps.
set -euo pipefail
cd "$(dirname "$0")/.."
[ "$(git branch --show-current)" = "main" ] || { echo "switch to main first"; exit 1; }
git fetch -q origin
for b in acer pc gpu-results; do
  if git rev-parse -q --verify "origin/$b" >/dev/null; then
    new=$(git rev-list --count "HEAD..origin/$b")
    if [ "$new" -gt 0 ]; then git merge -q --no-edit "origin/$b" && echo "merged origin/$b ($new commit(s))"; fi
  fi
done
python3 scripts/update_history_manifest.py
git add visualizer/frontend/research/history-manifest.json
git diff --cached --quiet || git commit -q -m "List new runs in the visualizer history"
git status -sb | head -1
