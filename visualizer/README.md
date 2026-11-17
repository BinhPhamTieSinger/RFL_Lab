# RFL Lab Visualizer

A local research workbench for the ten methods registered by RFL Lab. Python executes the existing research runner in a dedicated process. Next.js displays its recorded events over SSE.

## Run locally

From the repository root `Robust_Federated_Learning`, install backend dependencies into the existing environment.

```bash
.venv/bin/python -m pip install -r RFL_Lab/visualizer/backend/requirements.txt
cd RFL_Lab/visualizer/frontend
npm ci
```

Start the backend from `Robust_Federated_Learning`.

```bash
PYTHONPATH=RFL_Lab/visualizer/backend:RFL_Lab/src .venv/bin/python -m uvicorn rfl_viz.api:app --host 127.0.0.1 --port 8000
```

In another terminal, start the frontend.

```bash
cd RFL_Lab/visualizer/frontend
npm run dev
```

Alternatively, after installation run `bash RFL_Lab/visualizer/scripts/dev.sh` from the research root to start both services.

Open http://localhost:3000. The frontend proxies `/api` to the loopback backend. Keep a single API worker. The compute lock prevents simultaneous experiments across API processes. Application data is stored in `visualizer/data`, which is Git-ignored. `RFL_VIZ_DATA` selects another persistence directory.

## AI advisor (Documentation tab)

"Ask the docs (AI)" opens an advisor that answers questions about the documentation with your own Ollama models, citing the passages it used; each citation opens that document at that section. It runs in the browser: the documents are split at their headings, embedded with `bge-m3`, the six passages closest to the question are retrieved, and the answer model (default `qwen3.8:27b`, any installed model can be chosen in Settings) answers from them. Nothing is sent anywhere except your Ollama server. The embeddings are cached in the browser, so only the first question after a documentation change re-reads the documents. "Explain selection" turns text you have highlighted in a document into a question.

- **Visualizer on your machine** (`npm run dev`, http://localhost:3000): works as soon as the Ollama app is running; Ollama accepts requests from `localhost` pages by default.
- **Deployed site** (https://rfl-lab-visualizer.netlify.app): Ollama must also accept that page. On macOS, quit Ollama, run `launchctl setenv OLLAMA_ORIGINS "https://rfl-lab-visualizer.netlify.app"`, and open Ollama again (on Linux: `OLLAMA_ORIGINS=https://rfl-lab-visualizer.netlify.app ollama serve`). Use Chrome or Firefox; Chrome may ask to allow access to devices on your local network. Safari does not let an https page call `http://localhost`.
- **Ollama on another machine:** set its address in Settings (for example `http://192.168.1.20:11434`), start Ollama there with `OLLAMA_HOST=0.0.0.0` and the page's origin in `OLLAMA_ORIGINS`.

## Execution

The default runs one CPU round on 600 actual CIFAR-100 training images and 300 test images. Data comes from `Paper_Reproduction/data`. Configure additional rounds explicitly. Synthetic data is the research package's test fixture and is labeled as such in the dataset selector. Attacker identities follow the original seeded count or rate selection. Explicit arbitrary client IDs are unavailable in the original configuration.

The topology appears after actual model initialization. Client inspection exposes label histograms, sample counts, last local-training results and screening evidence. FedHEP clients upload directly to Global and their parent. Parent heads feed subsequent client training. The implementation performs in-process tensor transfers.

## Methods

| Method | Local rule and evaluation |
| --- | --- |
| FedAvg | SGD, sample-weighted Global mean, Global evaluation including attackers |
| FedProx | SGD with proximal penalty, Global evaluation including attackers |
| SCAFFOLD | Corrected gradients and control variates, Global evaluation including attackers |
| Multi-Krum | SGD and selection by Krum score, Global evaluation including attackers |
| Ditto | Shared update and personalized proximal training, honest personal-model evaluation |
| FedRep | Private classifier and shared representation, honest personal-model evaluation |
| FedHEP | Three heads, similarity grouping, parent momentum, Global mean |
| Defended FedHEP | Fixed groups, parent SCCF, Global SCCF, temporal trust ledger |
| FedHEP+ | Screening, z-scores, evidence, optional regrouping, parent SCCF and configured Global rule |
| Robust FedHEP | Configured Global robust rule followed by parent SCCF over accepted members |

FedHEP variants evaluate honest clients using their most recently trained local ensemble. Wrongly isolated honest clients remain evaluated. The research result's `evaluation_scope` string currently says honest clients for all methods, although the consensus source includes attackers for non-personalized methods. Use the method-specific behavior above.

## Inspection and replay

Select an execution event to inspect the accumulated state at that event. Return to live resumes following new events. History replays saved events and exposes original settings and final metrics. Mathematical inspection renders implementation explanations with source-file access. The formulas themselves are explanatory, while values in the execution inspector are captured.

Elapsed event time uses the worker's monotonic clock. Operation durations measure the wrapped Python call. GPU operations can enqueue asynchronous work, so the original runner's synchronized round resource timings are the authoritative GPU round measurement. Payload bytes describe flat tensor size and exclude protocol overhead.

## Boundaries

Per-class test accuracy and test loss are unavailable in the current runner. Checkpoint resume is unavailable. Cancellation terminates the isolated worker, retaining partial logs and events. A killed training kernel may require the ten-second escalation to process kill. API restart marks unfinished records interrupted. No automatic resume is attempted.

The interface renders the instantiated ResNet-9 module tree and first captured forward operations for client 0, round 1. Client inspectors expose captured forward shapes, output means, backward gradient norms, and optimizer measurements. Fine-grained activation tensor browsing and editable model graphs remain outside this implementation. Screening evidence and TTT decisions appear as their actual events arrive. Full round diagnostics remain available at round completion. Curated research history has its own manually edited manifest.

This service is designed for loopback use. Remote multi-user deployment requires authentication, authorization and production process supervision.

## Validation

```bash
PYTHONPATH=RFL_Lab/visualizer/backend:RFL_Lab/src .venv/bin/python -m pytest -q RFL_Lab/visualizer/tests
cd RFL_Lab/visualizer/frontend
npm run build
```

Tests compare observed and unobserved executions for every registered method with a smaller ResNet-9 width and real training on the synthetic fixture. API tests execute a standard-width training run, verify persistence, replay, cancellation, queued execution and configuration errors. See `docs/VALIDATION.md` for recorded execution evidence and remaining limitations.

## Detailed rounds and research history

The round walkthrough compares honest and malicious clients using actual partition histograms, CIFAR-100 thumbnails, R_skew, captured model operations, attack events and uploads. Phase controls focus the topology on clients, parents or Global. FedHEP+ screening is shown at Global before parent aggregation. Mathematics expands alongside the relevant phase.

Experiment history publishes the research-output folders selected by hand in `frontend/research/history-manifest.json`. Expand an entry to inspect its own charts, tables and configuration. Load configuration preserves supported fields and selects CPU. History refreshes every 15 minutes. New local training remains streamed continuously.

The hosted Live Workbench provides animated playback of real CPU captures for all ten models. Visualization captures are separate from research history. New training runs locally, as selected by the owner. See `docs/DEPLOYMENT.md` for GitHub-triggered publication and hosting details.

See `docs/PLAYBACK.md` for animation controls and ACLM/SCCF/TTT inspection, and `docs/HISTORY_CURATION.md` for maintaining research history. The Documentation tab renders the repository Markdown files with tables, mathematical notation and figures.
