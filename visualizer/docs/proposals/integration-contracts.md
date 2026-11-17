# RFL Lab Visualizer: integration contracts

Status: **proposal for review**, contract version `1.0.0-draft`, 2026-11-15. This is the interface between the backend (Claude) and the frontend (Astra). Architecture: [`visualizer-architecture.md`](visualizer-architecture.md). Event payloads: [`event-schema.md`](event-schema.md).

Once frozen, the machine-readable versions are the source of truth: the OpenAPI document served at `/openapi.json` and the JSON Schemas in `visualizer/backend/schemas/`. This document explains them.

## 1. Conventions

| Topic | Rule |
|---|---|
| Base URL | `http://127.0.0.1:8765/api/v1` (local only) |
| Format | JSON, UTF-8. Field names `snake_case`. |
| Versioning | Every response carries header `RFL-Viz-Contract: 1.0.0`. Additive changes (new optional fields, new event types) bump the minor version; anything else bumps the major version and the URL prefix. The frontend must ignore unknown fields and unknown event types. |
| Identifiers | Experiment id: `YYYYMMDD-HHMMSS-<method>-<4 hex>`, sortable by start time. Client ids are integers `0..N-1` as in the research code. Parent ids are integers `0..K-1`. |
| Component ids | `"global"`, `"parent:<g>"`, `"client:<c>"`, `"server"` (server-side work that belongs to no tier, such as FLTrust's root training). |
| Time | `t`: seconds since the worker started, float (monotonic, `perf_counter`). `wall_time`: ISO 8601 UTC, for display only. Durations: `duration_s`, seconds, float. |
| Numbers | Floats as JSON numbers; `NaN` and infinities are sent as `null` with a sibling `*_nonfinite: true`. Accuracy and success rates are percentages (0–100) as in `metrics.json`; cosines in [−1, 1]. |
| Provenance | Every displayed value is `observed`, `recorded` or `recomputed` (see event schema). The frontend shows the provenance on request (for example on hover in the inspector). |
| Null | `null` means "not applicable to this method or not available", never zero. |

## 2. Resources

### 2.1 Method
```json
{ "key": "fedhep_plus", "family": "fedhep_plus", "label": "FedHEP+",
  "description": "Defended FedHEP with robust z-score screening, permanent isolation and optional regrouping.",
  "model": "resnet9_multihead", "hierarchical": true }
```

### 2.2 Method schema (the configuration form)
Generated from the research configuration dataclasses. Only the sections the method reads are included.
```json
{ "method": "fedhep_plus",
  "sections": [
    { "key": "data", "label": "Data", "fields": [
      { "key": "data.num_clients", "type": "integer", "default": 15, "min": 2, "max": 100,
        "help": "Number of clients.", "source": "rfl.experiment.config.DataConfig.num_clients" },
      { "key": "data.alpha", "type": "number", "nullable": true, "default": null, "exclusive_min": 0,
        "help": "Dirichlet alpha; null = IID.", "null_label": "IID" } ] },
    { "key": "plus", "label": "FedHEP+ screening", "fields": [
      { "key": "plus.reference", "type": "enum", "options": ["mean_direction", "median"], "default": "mean_direction" } ] } ],
  "constraints": [
    { "rule": "attack.kind in [alie, min_max] requires a FedHEP-family method" },
    { "rule": "attack.count <= data.num_clients" } ] }
```
Field `type` is one of `integer`, `number`, `boolean`, `enum`, `string`. `help` is the comment from the dataclass. Hidden from the form: `run.output_dir` (set by the backend), `data.root` (set by the backend from the machine's data location).

### 2.3 Preset
A configuration taken unchanged from `configs/` or from one combination of a sweep file.
```json
{ "id": "sweep:phase6_ladder_d1:p6__r2__sign_flip", "label": "Phase 6 ladder R2, sign flip (D1)",
  "origin": "configs/sweeps/phase6_ladder_d1.yaml", "method": "fedhep_plus",
  "config": { "...": "full ExperimentConfig as nested JSON" },
  "research_runs": ["outputs/p6__r2__sign_flip_s42", "..."] }
```
`research_runs` lists existing research results for the same configuration, so the frontend can offer a comparison.

### 2.4 Topology specification
Declared by the method's adapter. The frontend draws exactly these node kinds and edges.
```json
{ "family": "fedhep_plus",
  "levels": [
    { "kind": "global", "ids": ["global"] },
    { "kind": "parent", "ids": ["parent:0", "parent:1", "parent:2"] },
    { "kind": "client", "ids": ["client:0", "...", "client:14"] } ],
  "edges": [
    { "from": "client:*", "to": "parent:<assignment>", "kind": "upload_parent" },
    { "from": "client:*", "to": "global", "kind": "upload_global" },
    { "from": "parent:*", "to": "client:<members>", "kind": "broadcast_parent_head" },
    { "from": "global", "to": "client:*", "kind": "broadcast_global" } ],
  "dynamic": { "assignment": "changes on grouping_step events" } }
```
Note: in the FedHEP family, clients upload directly to Global *and* to their parent (two separate uploads); parents do not forward to Global. The consensus family has two levels only (`global`, `client`).

### 2.5 Phase specification
```json
{ "family": "fedhep_plus", "phases": [
  { "id": "client_training", "label": "Client training", "components": "client:*" },
  { "id": "coordinated_attack", "label": "Coordinated attack", "when": "attack.kind in [alie, min_max]" },
  { "id": "sentinel", "label": "Non-finite check" },
  { "id": "screening", "label": "Screening" },
  { "id": "evidence", "label": "Evidence and isolation" },
  { "id": "grouping", "label": "Grouping step", "when": "plus.regroup" },
  { "id": "parent_step", "label": "Parent aggregation", "components": "parent:*" },
  { "id": "global_step", "label": "Global aggregation", "components": "global" },
  { "id": "evaluation", "label": "Evaluation", "when": "round % run.eval_every == 0 or last round" } ] }
```
`when` explains when a phase occurs; the backend only emits `phase_started` for phases that actually run, so the frontend does not evaluate `when` itself.

### 2.6 Experiment
```json
{ "id": "20261115-224107-fedhep_plus-3f2a", "method": "fedhep_plus", "state": "running",
  "round": 4, "rounds": 5, "phase": "screening",
  "created": "2026-11-15T15:41:07Z", "started": "2026-11-15T15:41:09Z", "finished": null,
  "config": { "...": "ExperimentConfig" },
  "environment": { "device": "mps", "torch": "2.x", "rfl_commit": "3db34b8", "rfl_dirty": false,
                   "machine": "Mac", "precise_timing": true, "concurrent_training_detected": false },
  "last_seq": 1842, "error": null, "observation_complete": true }
```
States: `validating`, `rejected`, `starting`, `running`, `cancelling`, `completed`, `failed`, `cancelled`, `interrupted`. Terminal states: the last five except `cancelling`.

### 2.7 Formula
```json
{ "id": "robust_z", "label": "Robust z-score",
  "latex": "z_i = \\frac{s_i - \\operatorname{med}(s)}{\\max(1.4826\\,\\mathrm{MAD}(s),\\ \\rho|\\operatorname{med}(s)|,\\ \\varepsilon)}",
  "variables": [ { "symbol": "s_i", "meaning": "screening score of client i (cosine to the reference)" } ],
  "source": { "function": "rfl.defense.screening.robust_z", "file": "src/rfl/defense/screening.py", "line": 173 },
  "families": ["fedhep_plus"],
  "bindings": { "event": "screening_z", "inputs": ["scores", "median", "mad", "relative_floor", "absolute_floor"],
                "output": "z" },
  "check": "rfl_viz.formulas.checks.robust_z" }
```
`source.line` is computed at start-up with `inspect`, so it follows the code. The frontend renders `latex` with KaTeX and fills variables from the bound event.

### 2.8 Artifact
```json
{ "id": "samples_client_3", "kind": "image_grid", "media_type": "image/png",
  "description": "16 training images of client 3 (after attack, as trained on)", "url": "/api/v1/experiments/<id>/artifacts/samples_client_3.png" }
```

## 3. Endpoints

| Method and path | Request | Response | Errors |
|---|---|---|---|
| `GET /health` | – | `{status, contract, rfl_commit, device, data_found}` | – |
| `GET /methods` | – | `Method[]` | – |
| `GET /methods/{key}/schema` | – | `MethodSchema` | 404 unknown method |
| `GET /methods/{key}/topology?num_clients=&num_groups=` | – | `TopologySpec` (initial) | 404 |
| `GET /methods/{key}/phases` | – | `PhaseSpec` | 404 |
| `GET /presets?method=` | – | `Preset[]` | – |
| `GET /formulas?family=` | – | `Formula[]` | – |
| `POST /experiments/validate` | `{method, config}` | `{valid: true, summary: {attackers, client_sizes, num_parents, estimated_seconds_per_round}}` | 422 field errors |
| `POST /experiments` | `{method, config, options: {precise_timing, per_batch}}` | 201 `Experiment` | 409 another experiment running; 422 invalid |
| `POST /experiments/{id}/cancel` | – | 202 `Experiment` (state `cancelling`) | 404; 409 already terminal |
| `GET /experiments?limit=&before=&method=&origin=visualizer\|research` | – | `Experiment[]` (summaries) | – |
| `GET /experiments/{id}` | – | `Experiment` | 404 |
| `GET /experiments/{id}/metrics` | – | `metrics.json` as written by the research code | 404; 409 not finished |
| `GET /experiments/{id}/events?from_seq=` | – | SSE stream (section 4) | 404 |
| `GET /experiments/{id}/events.jsonl` | – | full event log, newline-delimited JSON | 404 |
| `GET /experiments/{id}/stdout` | – | `stdout.log` as text | 404 |
| `GET /experiments/{id}/artifacts` | – | `Artifact[]` | 404 |
| `GET /experiments/{id}/artifacts/{name}` | – | file | 404 |

`estimated_seconds_per_round` comes from the median `seconds_per_round` of research runs of the same method on the same machine, and is labelled as an estimate.

Research runs imported into history (`origin=research`) have `metrics.json` and no events; the frontend shows their metrics and says that no event log exists for them.

## 4. Streaming (Server-Sent Events)

**Endpoint:** `GET /api/v1/experiments/{id}/events?from_seq=<n>` with `Accept: text/event-stream`.

**Framing:** one SSE message per event.
```
id: 1842
event: aggregation_completed
data: {"v":1,"experiment":"...","seq":1842,"t":37.512,"type":"aggregation_completed",...}

```
- `id` is the event's `seq`; the browser's `EventSource` sends it back as `Last-Event-ID` on reconnection, and the server resumes after it. `from_seq` does the same for a first connection (use `0` for a full replay).
- A comment line `: keep-alive` is sent every 15 seconds.
- After the terminal event (`experiment_completed`, `experiment_failed`, `experiment_cancelled` or `experiment_interrupted`), the server sends `event: end` and closes the stream.
- Output lines (stdout, stderr) travel in the same stream as events of type `stdout` and `stderr`, so their order relative to events is preserved.

**Guarantees**
1. Events of one experiment arrive in strictly increasing `seq`, without gaps, whether replayed or live.
2. A reconnecting client receives every event after its last `seq` exactly once.
3. Replay and live delivery produce the same sequence (the server writes each event to `events.jsonl` before broadcasting it).
4. The server never drops events for a slow client; it streams from the file at the client's pace.

**Rate (estimate, to be measured):** on the order of 100 events per round for FedHEP+ with 15 clients (two per client for training, one per attacking client, one per aggregation tier, phase markers, screening and evaluation) without per-batch events; with per-batch events, losses are sent once per client epoch in arrays, not once per batch.

## 5. Error model

All error responses use RFC 9457 problem details:
```json
{ "type": "https://rfl-lab/errors/invalid-config", "title": "Invalid configuration", "status": 422,
  "detail": "attack.count must be in [0, 15]",
  "errors": [ { "field": "attack.count", "message": "must be in [0, 15]", "source": "rfl.attacks.byzantine.select_attackers_fixed" } ] }
```
| `type` suffix | Status | Meaning |
|---|---|---|
| `invalid-config` | 422 | validation failed; `errors` lists fields |
| `experiment-running` | 409 | another experiment is running (includes its id) |
| `not-found` | 404 | unknown method, experiment or artifact |
| `already-finished` | 409 | cancel on a terminal experiment |
| `not-finished` | 409 | metrics requested before completion |
| `data-missing` | 503 | CIFAR-100 not found at the configured location |
| `internal` | 500 | backend bug; includes a reference to the log line |

Failures *during* training are not HTTP errors; they arrive as `experiment_failed` events and in `Experiment.error`.

## 6. Frontend obligations

1. **Never synthesize values.** Every number, state or relationship shown comes from an API response or an event. Interpolation between rounds, estimated progress bars without a backend value, and placeholder data in place of a missing value are not allowed; a missing value is shown as missing.
2. **Draw what is declared.** Topology from `TopologySpec`, phases from `PhaseSpec` and `phase_started` events. No method-specific layout in the frontend.
3. **Show provenance** on request for every value: observed, recorded or recomputed (with the function).
4. **Phase durations are boundary-to-boundary**, per-call durations are exact; label them accordingly.
5. **Ignore unknown fields and event types**, and show unknown event types in the terminal as raw JSON.
6. **Clearing the terminal clears the view only**; the stream and the log are unchanged.
7. **Reconnect** with `Last-Event-ID` (automatic with `EventSource`) and rebuild state by replay.
8. **State is a pure function of the event sequence** (a reducer), so a replay of `events.jsonl` reproduces exactly what was shown live.

## 7. Backend obligations

1. Every event validates against its JSON Schema before it is written.
2. No event is emitted for a step that did not execute.
3. Every `recomputed` value names the repository function used.
4. The research `metrics.json` of a visualizer run is identical to an unobserved research run with the same configuration and seed on the same machine.
5. Schemas, OpenAPI and fixture streams are versioned together; a breaking change is never shipped without a major version bump.

## 8. Development without a running backend

Astra develops against **recorded fixture streams**, not mocks: `visualizer/fixtures/streams/<family>__<attack>.jsonl`, each recorded from a real run on the synthetic dataset (CPU, a few rounds), together with its `metrics.json`. A small replay server (`python -m rfl_viz.replay <file> --speed 1.0`) serves a fixture over the same SSE endpoint at recorded or accelerated pace. Fixtures are regenerated whenever the event schema changes, and every fixture validates against the schemas in CI.

## 9. Change process

Contract changes are proposed as a pull request that updates this document, the OpenAPI document, the JSON Schemas and the fixtures together. Astra and Claude both approve; the reviewer (ChatGPT) is consulted for breaking changes.
