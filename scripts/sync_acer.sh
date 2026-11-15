#!/usr/bin/env bash
# Copy finished runs from the Acer into this checkout. Run on the Mac, from the repository root:
#   bash scripts/sync_acer.sh
# Every run has its own folder name, so nothing already on the Mac is touched (--ignore-existing); a run still in
# progress arrives without metrics.json and is completed by the next sync. Override host or path with ACER, ACER_DIR;
# ssh uses the Mac's SSH config for the key.
set -euo pipefail
HOST=${ACER:-tiesinger@100.117.221.19}
REMOTE=${ACER_DIR:-Documents/RFL_Lab}
rsync -av --ignore-existing "$HOST:$REMOTE/outputs/" outputs/
