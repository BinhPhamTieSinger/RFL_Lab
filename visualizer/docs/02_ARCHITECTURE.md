# Visualizer, Stage 2: architecture and implementation plan

Proposal, 2026-11-15. Builds on the inventory in [`01_REPOSITORY_INVENTORY.md`](01_REPOSITORY_INVENTORY.md). Nothing here is built yet except the observation spike that proves the central design choice.

## 1. Principles

1. **The research code runs unchanged.** The visualizer starts the same `run_experiment` the experiments use. It observes from outside with instance-level wrappers and the server's own diagnostics. A test runs every adapter twice, observed and unobserved, and requires identical `metrics.json` (already passing for three methods).
2. **Every displayed value has a source.** Each value is either *observed* (read from the running code), *recorded* (written by the research code itself: `diagnostics`, `metrics.json`, stdout) or *recomputed* (the same library function applied to observed inputs, labelled as such, with the function named). Nothing is estimated or animated without an event behind it.
3. **One adapter per structural family**, not per screen. The four families (consensus, FedHEP, Defended FedHEP, and its two descendants) have different phases; the frontend draws whatever phases the adapter declares.
4. **Python computes; the browser presents.** No algorithm is reimplemented in TypeScript.
5. **Simplest thing that works**: one FastAPI process, one worker process per experiment, files on disk for persistence. No database, queue server or microservices.

## 2. Process model

```
Browser (Next.js app)
   │  REST: models, schemas, start, cancel, history
   │  SSE:  /experiments/{id}/events  (structured events + stdout lines)
   ▼
FastAPI server (rfl_viz.api)            one process, async
   │  spawns, monitors, cancels
   ▼
Worker process (rfl_viz.worker)         one per experiment, multiprocessing "spawn"
   │  with observed(emit): run_experiment(cfg)     ← unchanged research code
   │  emit → multiprocessing.Queue → server → SSE + events.jsonl
   │  stdout → pipe → server → SSE + stdout.log
   ▼
experiments/<id>/  config.yaml · metrics.json (written by run_experiment) · events.jsonl · stdout.log · status.json
```

Why a separate process: training holds the GPU and the Python interpreter for minutes; a process keeps the API responsive, makes cancellation safe (terminate the process, mark the experiment cancelled, keep what was written) and isolates crashes. Why SSE rather than WebSocket: the stream is one-way; commands (start, cancel) are ordinary REST calls; SSE reconnects automatically and needs no extra protocol.

## 3. Backend modules (`visualizer/backend/rfl_viz/`)

| Module | Responsibility |
|---|---|
| `schema.py` | Builds each method's configuration form from the `rfl.experiment.config` dataclasses (types, defaults, comments) plus a relevance map (which sections and fields each method reads; inventory section 10). Validation is `load_config` itself plus the method/model and attack/method constraints the registry enforces, run before a worker starts. |
| `adapters/` | One adapter per family: `consensus.py`, `fedhep.py`, `defended.py` (also used by FedHEP+ and robust FedHEP with extensions). Each declares its **topology** (root, parents or not, clients), its **phases in code order**, the **wrappers** to install and how to turn `diagnostics` into events. |
| `observe.py` | The `observed(emit)` context manager (spike exists). |
| `worker.py` | Entry point of the worker process: builds the config, installs observation, captures stdout line by line, calls `run_experiment`, reports completion or the exception. |
| `orchestrator.py` | Starts, tracks and cancels workers; fans events out to SSE subscribers and to `events.jsonl`; writes `status.json`. One experiment at a time per machine by default (the GPU is the bottleneck). |
| `store.py` | Lists and reads past experiments from `experiments/`; can also import any existing run folder from `outputs/` (read-only history of the 1,100+ research runs, without events). |
| `api.py` | FastAPI routes (section 6). |

## 4. Event schema

Every event is one JSON object:

```json
{
  "v": 1,
  "experiment": "2026-11-15T22-41-07_fedhep_plus",
  "seq": 1842,
  "t": 37.512,
  "type": "aggregation_completed",
  "round": 4,
  "phase": "global_step",
  "component": "global",
  "duration_s": 0.041,
  "source": "observed",
  "data": { "tier": "global_defense", "senders": 13, "trust": [0.081, 0.077, "..."] }
}
```

`t` is seconds since the worker started (`time.perf_counter`), `duration_s` is measured around the wrapped call, `seq` orders events. `source` is `observed`, `recorded` or `recomputed` (with `data.function` naming the library function).

Event types, each emitted only by the families that execute that step:

| Type | Families | Payload (all from code) |
|---|---|---|
| `experiment_started` | all | full config, device, method, family, attackers |
| `data_partitioned` | all | per client: train/test sizes, label histogram (100 bins), $r_{\text{skew}}$ via `label_stats` (recomputed) |
| `round_started` / `round_completed` | all | round, duration |
| `phase_started` / `phase_completed` | all | phase id from the adapter's list |
| `client_training_started` / `_completed` | all | client, samples, poisoned labels; FedHEP family adds $r_{\text{skew}}$, head loss weights (recomputed with `binomial_head_weights`), last-batch losses per head, ensemble weights, optimizer steps |
| `attack_executed` | all with attacks | client, kind, clean vs sent update norm and cosine; coordinated attacks: observed set, crafted update norm, γ for Min-Max |
| `grouping_step` | FedHEP; FedHEP+ with regroup | event (none, initial, update), assignment, stability or misaligned count (recomputed) |
| `screening_scored` | FedHEP+ | per client: score, z, two-sided z, gap ratio, median, MAD, scale, floors (recorded) |
| `client_left_out` / `client_isolated` | FedHEP+, Defended FedHEP (ledger) | client, round, evidence or ledger score, reason |
| `aggregation_completed` | all | tier (parent g, Global), senders, rule, weights or trust per sender, accepted set |
| `model_updated` | all | tier, update norm |
| `evaluation_completed` | all | pooled, mean client, bottom-10%, per client accuracy, backdoor success overall and per head (recorded) |
| `detection_recorded` | methods with a report | flagged, precision, recall, F1, AUROC (recorded) |
| `stdout` | all | one line of the worker's output, verbatim |
| `experiment_completed` / `_failed` / `_cancelled` | all | final metrics path, or the exception and traceback |

## 5. Topology and phases per family

| Family | Topology | Phases (code order) |
|---|---|---|
| Consensus | Global ↔ clients | client training (with attack hooks) → aggregation → global update → evaluation |
| FedHEP | Global ↔ parents ↔ clients | client training → coordinated attack → grouping step → parent step (copy rule or mean, then momentum) → Global mean → evaluation |
| Defended FedHEP | same, groups fixed | client training → coordinated attack → sentinel → parent SCCF + ledger → Global SCCF + ledger → evaluation |
| FedHEP+ | same, groups from kept clients | client training → coordinated attack → sentinel → screening (scores, z, gate) → evidence and isolation → grouping (if regroup) → parent SCCF → Global SCCF or L0 → evaluation |
| Robust FedHEP | same, groups fixed | client training → coordinated attack → sentinel → (FLTrust: server training) → Global rule → parent SCCF over accepted → evaluation |

The frontend reads this list from the backend; it holds no family-specific layout.

## 6. API

| Route | Purpose |
|---|---|
| `GET /models` | methods, families, descriptions |
| `GET /models/{method}/schema` | configuration form for that method (relevant fields only, with defaults and constraints) |
| `POST /experiments/validate` | dry validation with the research code's own loader; returns errors or a summary (attackers, partition sizes, estimated rounds) |
| `POST /experiments` | start; returns id |
| `POST /experiments/{id}/cancel` | stop the worker |
| `GET /experiments` | history (visualizer runs and imported research runs) |
| `GET /experiments/{id}` | config, status, metrics |
| `GET /experiments/{id}/events` | SSE: replay from `seq` (so a reload loses nothing), then live |
| `GET /formulas` | formula catalogue: LaTeX, variables, source function, which event fields feed it |

## 7. Frontend (`visualizer/frontend/`)

Next.js (app router), TypeScript, Tailwind, shadcn/ui primitives (restyled to the monochrome system), React Flow for the topology, Recharts for metrics, KaTeX for formulas, xterm.js for the terminal, Framer Motion only for state transitions that carry meaning (a phase starting, a client being isolated).

| View | Content |
|---|---|
| Live Workbench | sidebar (schema-driven form, validation, summary, start/cancel) · canvas (topology, phase timeline, live state) · inspector (selected client, parent or Global: data, training, attack, scores, formulas with live values) · terminal (stdout and events, filters, copy, clear view without deleting logs) |
| Architecture | the selected family's phases and data flow, drawn from the adapter specification |
| Formulas | the catalogue, each linked to the source function |
| History | past experiments: config, metrics, replay of the event log |
| Documentation | the repository's docs rendered |

**Visual encoding without colour**: benign client = light filled node; malicious = hatched fill, heavier outline, "malicious" label; left out this round = dashed outline, "left out"; isolated = dashed outline, struck label, "isolated"; active component = strongest contrast; completed = subdued grey. Lines: solid = upload accepted, dashed = rejected or excluded, dotted = broadcast. Charts distinguish series by line style and marker, never by hue.

## 8. Design system plan (Stage 3)

Two themes, grayscale only. Dark: background #0b0b0b, surfaces #161616 and #1f1f1f, borders #2c2c2c, text #f2f2f2 and #a3a3a3, disabled #5c5c5c. Light: background and surfaces #ffffff, secondary #f4f4f4, borders #e2e2e2, text #141414 and #6b6b6b, disabled #a8a8a8. One typeface, IBM Plex Sans (IBM Plex Mono only in the terminal and code), KaTeX for mathematics. Spacing on a 4 px grid; one corner radius (4 px); one border weight. Before building screens, the three named design skills (Impeccable, UI UX Pro Max, draw.io) will be cloned from GitHub and their actual instructions read: UI UX Pro Max for the system, Impeccable for critique, draw.io for the architecture documentation. They are not installed in this environment today; their repositories are reachable.

## 9. Validation (Stage 7)

1. **Passive observation**: observed vs unobserved runs give identical `metrics.json`, for every method and attack kind (extends the spike's test to all ten methods).
2. **Event consistency**: per round, the clients in `client_training_completed` equal the method's clients; isolations in events equal `diagnostics`; aggregation weights sum to 1; the evaluation event equals the `metrics.json` row.
3. **Order**: each adapter's declared phases match the order in which wrapped calls actually fire (recorded `seq`).
4. **Recomputed values**: every recomputed quantity is checked against the recorded one where both exist (for example $r_{\text{skew}}$ recomputed vs `HeadMemory.r_skew`).
5. **End to end on the Mac with CIFAR-100**: one round, three rounds, five rounds of FedHEP+ under sign flip; the run must equal a research run of the same config and seed.

## 10. Implementation plan

| Stage | Output | Exit check |
|---|---|---|
| 1 Inventory | `01_REPOSITORY_INVENTORY.md` | done |
| 2 Architecture | this document; observation spike | spike: identical metrics for three methods (done) |
| 3 Design system | tokens, type scale, components, diagram and terminal styles; both themes; reviewed with the design skills | a static style sheet page in both themes |
| 4 Backend, one model end to end | schema, worker, orchestrator, SSE, store for **FedHEP+**; passive-observation test for it | a 3-round synthetic run and a 1-round CIFAR-100 run streamed and persisted |
| 5 Visualization for FedHEP+ | workbench with topology, timeline, inspector, formulas with live values, terminal | every displayed number traced to an event |
| 6 Other families | consensus, FedHEP, Defended FedHEP, robust FedHEP adapters and diagrams | passive-observation test for all ten methods |
| 7 Validation | section 9 | all checks pass |
| 8 Polish | critique with the design skills; accessibility; empty and error states | both themes reviewed screen by screen |

## 11. Decisions needed before Stage 3

1. **Per-batch losses.** The trainers keep only the last batch's loss per head. Showing every batch needs an opt-in callback in the trainers (default off, no change to any computation, covered by the passive-observation test). Without it the client panel shows last-batch losses and optimizer step counts only.
2. **Where it runs.** Training needs the CIFAR-100 files and a GPU, so the visualizer is meant to run on the Mac (or the Acer). It can be developed and tested here on the synthetic dataset and CPU; real runs happen on your machine.
3. **Scope of the first model.** FedHEP+ first, as in the brief's example, with the configuration the experiments actually use (median reference, regrouping), then the others.
