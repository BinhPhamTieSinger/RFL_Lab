# Deployment

The Netlify site is `https://rfl-lab-visualizer.netlify.app`.

New training stays on the owner's workstation. Run `bash visualizer/scripts/dev.sh` from RFL_Lab and open `http://127.0.0.1:3000`. CPU is the default, including configurations loaded from historical GPU runs. The local API serializes workers and supports queued and active cancellation.

Netlify runs with `RFL_ARCHIVE_ONLY=1`. It serves curated research results and captured CPU visualization traces. The Load visualization button opens a trace, and Play animation advances its actual events. New training is available through the local Python engine.

Continuous deployment uses the existing private GitHub repository, its main branch, a read-only Netlify deploy key and the GitHub push webhook. Build configuration is in `visualizer/frontend/netlify.toml`. The build runs `npm run build` in `visualizer/frontend`. The prebuild exporter reads the output folders selected in `frontend/research/history-manifest.json`. Additional local-only results present at initial publication are preserved in `frontend/research/additional-history.json`. New results need to be pushed to GitHub for hosted publication.

History fetches every 900,000 milliseconds while the page is open. A successful GitHub-triggered deployment makes newly pushed results available on the next refresh. Local active runs stream via SSE continuously. There is no automatic Git pull into the working checkout.

The visualization capture JSON files and additional history are intentional publication artifacts. Datasets, checkpoints, credentials and local SQLite databases remain excluded. Source excerpts and configuration schema snapshots are bundled in `frontend/research/catalog.json`. Regenerate this catalog when changing research implementations or the formula registry.

## Workbench playback and curated history

The hosted Live Workbench opens directly. Select a model's captured visualization, load it and use Play animation, Pause animation, Next operation or the event timeline. Inspection speed is adjustable. Recorded computation durations remain visible. Reduced-motion mode shows static transfer markers.

New training continues to use the local Python engine. Hosted playback uses captured real events. Every capture hydrates its own configuration before its topology appears, including attack settings. The hierarchy, attack roles and loaded sidebar always refer to the same execution.

History membership is now maintained by hand in `frontend/research/history-manifest.json`. See HISTORY_CURATION.md. Live visualization captures and local worker runs are excluded. The GitHub deployment rebuilds the selected history and the current Markdown documentation.

ACLM captures parent/local class masking. SCCF records actual clipped norms, cosines and trust weights. TTT records before/after ledger values for Defended FedHEP. FedHEP+ uses screening evidence instead. Transfer animations represent local tensor uploads and aggregation inputs. The implementation sends client uploads directly to Global and parents, so the visualization follows those paths.
