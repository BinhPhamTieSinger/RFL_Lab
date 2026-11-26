#!/usr/bin/env bash
# Runner side of the branch workflow (docs/OPERATIONS.md, section 4). Each runner owns one branch on GitHub:
# the Acer pushes to "acer", the GPU server to "pc". The Mac merges both into main.
#
#   bash scripts/runner_sync.sh acer            # commit finished runs and push them (safe while a sweep runs)
#   bash scripts/runner_sync.sh acer --update   # also take the newest code and sweep files from main
#                                               # (only between sweeps: a running sweep keeps its old code)
set -euo pipefail
BRANCH=${1:?usage: runner_sync.sh <acer|pc> [--update]}
cd "$(dirname "$0")/.."
if [ "$(git branch --show-current)" = main ]; then   # the Mac's hub branch: running this there renames main and
  echo "on main: this is the hub (use scripts/hub_collect.sh); a runner renames its branch first (OPERATIONS.md 4)"
  exit 1                                             # force-pushes it over the runner's branch on GitHub
fi
[ "$(git branch --show-current)" = "$BRANCH" ] || git branch -M "$BRANCH"
for d in outputs outputs_parity outputs_env outputs_pilot; do
  if [ -d "$d" ]; then git add "$d"; fi
done
git diff --cached --quiet || git commit -q -m "$BRANCH results $(date +%Y-%m-%d_%H%M)"
git fetch -q origin
if [ "${2:-}" = "--update" ]; then
  git rebase --autostash origin/main      # uncommitted code edits are set aside and put back
fi
git push -q --force-with-lease origin "HEAD:$BRANCH"
echo "$BRANCH: $(git log -1 --format='%h %s')"
echo "runs on this machine: $(ls outputs | wc -l); behind main by $(git rev-list --count HEAD..origin/main) commit(s)"
