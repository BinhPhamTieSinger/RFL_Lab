# Integration and event contract

`rfl.experiment.registry.METHODS` supplies the ten supported methods. `catalog.py` derives configuration fields from the research dataclasses and filters model-specific sections. `observe.py` wraps server, trainer, attack and aggregator instances in the worker process. Its context restores every wrapped attribute. Research source files remain untouched.

`worker.py` calls the original `run_experiment`. The supervisor drains actual stdout and stderr together, stores log lines and structured events in SQLite, then records the exit status. Training uses two CPU threads by default. A file lock spans worker execution. UUID experiment paths contain configuration and results. SQLite assigns monotonic event IDs. SSE replays by `Last-Event-ID` or the `after` query parameter.

## API

- `GET /models` and `GET /models/{method}/schema` expose method capabilities and configuration.
- `POST /experiments/validate` checks a method and dotted overrides.
- `POST /experiments` launches a validated experiment and returns its ID.
- `GET /experiments` and `GET /experiments/{id}` expose persisted history and results.
- `POST /experiments/{id}/cancel` terminates the owned worker.
- `GET /experiments/{id}/events` replays and streams events.
- `GET /formulas`, `GET /source/{path}` and `GET /documentation` support inspection.

## Event schema v1

Every persisted event contains `v`, `experiment`, Unix `timestamp` and `type`. SSE `id` is the SQLite sequence. Worker events additionally carry monotonic `elapsed_s`. Round events identify `round`. Client events identify `client`. `duration_s` measures completed wrapped calls.

Initialization records actual clients, class histograms, attackers, groups, parent IDs, parameter count and module representation. Training start/completion events bracket the real trainer call. First-step optimizer events expose actual learning rates and the first twelve parameter gradient norms. Module hooks record the first forward output shape/mean and backward gradient norm per module in each client training call. All optimizer steps are counted. This bounded capture avoids serializing model tensors.

Upload-ready events describe completed in-process tensor production and actual payload size. They do not assert packet transmission. Aggregation events identify the actual invoked tier, with parent ID and members where available. Robust-rule events expose accepted indices and scores. Screening events expose the original cosine-score function's output. Isolation events follow changes in the original evidence tracker. Round diagnostics copy the server's recorded result. Evaluation events copy the original AccuracyReport.

Non-finite floating-point values are represented by their Python string (`inf`, `-inf`, `nan`) so JSON remains valid while retaining the numerical condition. Final result metadata also uses this representation.

## Source inventory

- `experiment/config.py`, `registry.py`, `runner.py` define configuration, component construction and lifecycle.
- `models/resnet9.py` defines the single-head and three-head ResNet-9 variants.
- `data/cifar100.py`, `clients.py`, `partition.py` define normalization, seeded subset selection and IID/Dirichlet distributions.
- `client/sgd.py`, `fedprox.py`, `scaffold.py`, `ditto.py`, `fedrep.py`, `fedhep.py` define actual local rules.
- `server/fedavg_server.py` handles the six consensus methods.
- `server/fedhep_server.py`, `defended_fedhep_server.py`, `fedhep_plus_server.py`, `robust_fedhep_server.py` define the hierarchical variants and ordering.
- `attacks/byzantine.py`, `backdoor.py`, `adaptive.py` implement data, label, independent-upload and coordinated attacks.
- `defense/screening.py`, `soft_cosine.py`, `trust_ledger.py`, `classic.py` implement the inspection targets.
- `grouping/similarity.py` defines seeded random assignment and optional update-based regrouping.
- `evaluation/accuracy.py`, `backdoor.py`, `detection.py` define reported metrics.

The root initialization copies the initial state into every parent. Clients train from Global plus their parent's classifier and their own remembered classifier. FedHEP+ screens after uploads, updates evidence, optionally regroups and aggregates parents before Global. Robust FedHEP computes its Global rule first, then updates parents with accepted clients. Flat methods never instantiate parents.
