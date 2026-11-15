# Deployment

The Netlify site is `https://rfl-lab-visualizer.netlify.app`.

New training stays on the owner's workstation. Run `bash visualizer/scripts/dev.sh` from RFL_Lab and open `http://127.0.0.1:3000`. CPU is the default, including configurations loaded from historical GPU runs. The local API serializes workers and supports queued and active cancellation.

Netlify runs with `RFL_ARCHIVE_ONLY=1`. It serves saved research results and the labeled, captured CPU smoke-round transcript. The Run button explains that training runs locally. It never starts a simulated experiment.

Continuous deployment uses the existing private GitHub repository, its main branch, a read-only Netlify deploy key and the GitHub push webhook. Build configuration is in `visualizer/frontend/netlify.toml`. The build runs `npm run build` in `visualizer/frontend`. The prebuild exporter reads all `outputs*/**/config.yaml` and `metrics.json` files. Additional local-only results present at initial publication are preserved in `frontend/research/additional-history.json`. New results need to be pushed to GitHub for hosted publication.

History fetches every 900,000 milliseconds while the page is open. A successful GitHub-triggered deployment makes newly pushed results available on the next refresh. Local active runs stream via SSE continuously. There is no automatic Git pull into the working checkout.

The recorded round JSON and additional history are intentional research-result publication artifacts. Datasets, checkpoints, credentials and local SQLite databases remain excluded. Source excerpts and configuration schema snapshots are bundled in `frontend/research/catalog.json`. Regenerate this catalog when changing research implementations or the formula registry.
