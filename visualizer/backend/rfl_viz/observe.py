"""Process-local passive adapters. No updates, RNG draws or training operations are replaced."""

import contextlib
import time
from dataclasses import asdict

import torch

from rfl.experiment import registry, runner
from rfl_viz.partitions import describe_clients


@contextlib.contextmanager
def observed(emit):
    undo = []
    state = {"round": 0, "client": None, "training": False, "steps": 0, "parent": None}

    def wrap(obj, name, before=None, after=None):
        original = getattr(obj, name)

        def call(*args, **kwargs):
            if before:
                before(*args, **kwargs)
            begin = time.perf_counter()
            result = original(*args, **kwargs)
            if after:
                after(result, time.perf_counter() - begin, *args, **kwargs)
            return result

        setattr(obj, name, call)
        undo.append(lambda: setattr(obj, name, original))

    def event(event_type, **data):
        emit({"type": event_type, "round": state["round"], **data})

    def instrument(server, cfg, model, clients):
        hierarchical = hasattr(server, "grouping")
        event(
            "initialized",
            method=cfg.method,
            attackers=sorted(server.attackers),
            groups=dict(server.grouping.assignment) if hierarchical else {},
            parents=list(server.parents) if hierarchical else [],
            architecture=str(model),
            parameters=sum(p.numel() for p in model.parameters()),
            clients=describe_clients(cfg, clients),
            dataset=cfg.data.dataset,
            attack=asdict(cfg.attack),
            configured_rounds=cfg.run.rounds,
            configuration={
                "method": cfg.method,
                "overrides": {
                    f"{section}.{key}": value
                    for section, values in asdict(cfg).items()
                    if isinstance(values, dict)
                    for key, value in values.items()
                },
            },
        )
        # Leaf-module hooks capture one actual forward per module per client training call.
        seen_modules = set()

        def forward_hook(name):
            def capture(module, inputs, output):
                if not state["training"] or name in seen_modules or not isinstance(output, torch.Tensor):
                    return
                seen_modules.add(name)
                event(
                    "operation_completed",
                    client=state["client"],
                    operation="forward",
                    module=name,
                    module_type=type(module).__name__,
                    shape=list(output.shape),
                    output_mean=float(output.detach().mean()),
                )
                if output.requires_grad:

                    def gradient(grad):
                        event(
                            "operation_completed",
                            client=state["client"],
                            operation="backward",
                            module=name,
                            gradient_norm=float(grad.detach().norm()),
                        )
                        return grad

                    output.register_hook(gradient)

            return capture

        observed_model = server.model if hierarchical else server.work_model
        for name, module in observed_model.named_modules():
            if name and not list(module.children()):
                handle = module.register_forward_hook(forward_hook(name))
                undo.append(handle.remove)
        memories = {id(m): i for i, m in enumerate(server.memory)} if hierarchical else {}

        def round_start(r):
            state["round"] = r
            state["initial_global"] = server.global_vec.clone() if hierarchical else None
            event("round_started")

        def round_end(result, seconds, r):
            runner._sync(clients[0].x_train.device)
            if hierarchical:
                event(
                    "model_updated",
                    tier="global",
                    update_norm=float((server.global_vec - state["initial_global"]).norm()),
                )
            if getattr(server, "diagnostics", None):
                event("diagnostics", values=server.diagnostics[-1])
            if server.report:
                from rfl.evaluation.detection import detection_metrics

                event("detection", values=detection_metrics(server.report, server.attackers, len(clients)))
            event("round_completed", duration_s=seconds)

        wrap(server, "run_round", round_start, round_end)

        def train_start(model, data, r, identity, targets=None):
            cid = memories.get(id(identity), "server_reference") if hierarchical else identity
            state["client"] = cid
            seen_modules.clear()
            state["training"] = True
            state["steps"] = 0
            state["mask_calls"] = 0
            state["num_classes"] = server.trainer.num_classes if hierarchical else 0
            state["last_loss"] = None
            event("client_training_started", client=cid, samples=data.num_train)

        def train_end(result, seconds, model, data, r, identity, targets=None):
            state["training"] = False
            payload = {
                "duration_s": seconds,
                "client": state["client"],
                "optimizer_steps": state["steps"],
                "last_backward_objective": state["last_loss"],
            }
            if hierarchical:
                payload.update(
                    head_losses=dict(identity.head_losses),
                    r_skew=identity.r_skew,
                    ensemble_alpha=list(identity.ensemble_alpha),
                )
            event("client_training_completed", **payload)

        wrap(server.trainer, "train", train_start, train_end)
        if hierarchical:

            def alpha_updated(result, seconds, alpha, prior, z_local, z_parent, z_root, y, lr):
                if state["steps"] == 1:
                    event(
                        "ensemble_mixing_updated",
                        client=state["client"],
                        input_alpha=alpha.tolist(),
                        prior=prior.tolist(),
                        output_alpha=result.tolist(),
                        learning_rate=lr,
                        alpha_lr_scale=server.trainer.alpha_lr_scale,
                        prior_mix=server.trainer.prior_mix,
                        duration_s=seconds,
                        scope="first batch, actual _alpha_step inputs and output",
                    )

            wrap(server.trainer, "_alpha_step", after=alpha_updated)

        if hierarchical:

            def mask_members(result, seconds, members):
                state["members"] = list(members)

            if hasattr(server, "_masks"):
                wrap(server, "_masks", after=mask_members)

            if hasattr(server, "ledger") and cfg.method == "fedhep_defended":

                def trust_before(trust, num_senders):
                    state["trust_before"] = dict(server.ledger.score)

                def trust_after(result, seconds, trust, num_senders):
                    event(
                        "temporal_trust_updated",
                        tier=state.get("tier"),
                        parent=state.get("parent"),
                        members=list(trust),
                        trust=dict(trust),
                        num_senders=num_senders,
                        before={c: state["trust_before"][c] for c in trust},
                        after={c: server.ledger.score[c] for c in trust},
                        momentum=server.ledger.momentum,
                        threshold=server.ledger.threshold,
                        isolated=sorted(server.ledger.isolated),
                        duration_s=seconds,
                    )

                wrap(server.ledger, "update", trust_before, trust_after)

            def members(result, seconds, group, among):
                state["parent"] = group
                state["members"] = result
                state["pending_parent"] = group if result else None

            wrap(server.grouping, "members", after=members)
            if hasattr(server, "evidence"):
                previous = set()

                def isolation(result, seconds, *args):
                    nonlocal previous
                    event(
                        "screening_updated",
                        tier="global",
                        z=dict(args[0]),
                        evidence=dict(server.evidence.evidence),
                        isolated=sorted(server.evidence.isolated),
                        evidence_cut=server.evidence.cut,
                        decay=server.evidence.decay,
                        patience=server.evidence.patience,
                        duration_s=seconds,
                    )
                    for cid in server.evidence.isolated - previous:
                        event("client_isolated", client=cid, evidence=server.evidence.evidence.get(cid))
                    previous = set(server.evidence.isolated)

                wrap(server.evidence, "update", after=isolation)

            def uploads(result, seconds, start, r):
                for tier, values in zip(("global", "parent"), result, strict=True):
                    for cid, vec in values.items():
                        event(
                            "upload_ready",
                            client=cid,
                            tier=tier,
                            update_norm=float((vec - start).norm()),
                            bytes=vec.numel() * vec.element_size(),
                            transport="in-process tensor",
                        )

            wrap(server, "_train_all", after=uploads)
            wrap(
                server.grouping,
                "step",
                after=lambda result, seconds, *args: event(
                    "grouping_completed",
                    groups=dict(server.grouping.assignment),
                    decision=result.value,
                    duration_s=seconds,
                ),
            )
        if cfg.method == "fedhep":
            from rfl.server import fedhep_server

            def agreement(result, seconds, *args):
                state["parents_copy_global"] = result > server.sync_similarity

            wrap(fedhep_server, "mean_member_similarity", after=agreement)

            def average_start(members, deltas):
                state["average_tier"] = (
                    "parent"
                    if state.get("pending_parent") is not None and not state.get("parents_copy_global")
                    else "global"
                )
                state["pending_parent"] = None

            wrap(server, "_average", before=average_start)

        for attr in ("aggregator", "parent_defense", "global_defense"):
            if hasattr(server, attr):

                def begin(*args, _tier=attr, **kw):
                    state["tier"] = (
                        state.get("average_tier", "global")
                        if _tier == "aggregator"
                        else "parent"
                        if _tier == "parent_defense"
                        else "global"
                    )
                    state["aggregating"] = True
                    state["cosines"] = []
                    info = {
                        "tier": state["tier"],
                        "algorithm": "SCCF"
                        if _tier.endswith("defense")
                        else type(getattr(server, _tier)).__name__,
                    }
                    if _tier.endswith("defense"):
                        info["members"] = list(state.get("members", []))
                        info["input_norms"] = [float(update.norm()) for update in args[0]]
                    if _tier == "aggregator":
                        info["members"] = [update.client_id for update in args[0]]
                        info["input_norms"] = [float(update.delta["state"].norm()) for update in args[0]]
                        if not hierarchical:
                            for update in args[0]:
                                vector = update.delta["state"]
                                event(
                                    "upload_ready",
                                    client=update.client_id,
                                    tier="global",
                                    bytes=vector.numel() * vector.element_size(),
                                    update_norm=float(vector.norm()),
                                    transport="in-process tensor",
                                )
                    if state["tier"] == "parent":
                        info["parent"] = state.get("parent")
                    state["aggregation_context"] = info
                    event("aggregation_started", **info)

                def end(result, seconds, *args, _tier=attr, **kw):
                    info = {**state.get("aggregation_context", {}), "duration_s": seconds}
                    if isinstance(result, tuple):
                        info["trust"] = result[1].tolist()
                        info["output_norm"] = float(result[0].norm())
                    elif isinstance(result, dict) and "state" in result:
                        info["output_norm"] = float(result["state"].norm())
                    event("aggregation_completed", **info)
                    state["aggregating"] = False

                wrap(getattr(server, attr), "aggregate", begin, end)
        if hasattr(server, "_apply"):
            wrap(
                server,
                "_apply",
                before=lambda *a: event("aggregation_started", tier="global", rule=server.rule),
                after=lambda result, seconds, *a: event(
                    "aggregation_completed",
                    tier="global",
                    rule=server.rule,
                    accepted=result.accepted,
                    scores=result.scores.tolist(),
                    duration_s=seconds,
                ),
            )
        if server.attack:

            def attack_client(cid, data):
                state["client"] = cid

            wrap(server.attack, "poison_data", before=attack_client)
            kind = cfg.attack.kind
            if kind not in ("label_flip", "backdoor", "alie", "min_max"):
                wrap(
                    server.attack,
                    "poison_upload",
                    after=lambda result, seconds, trained, start: event(
                        "attack_executed",
                        client=state["client"],
                        kind=kind,
                        clean_norm=float((trained - start).norm()),
                        sent_norm=float((result - start).norm()),
                    ),
                )
            if kind == "label_flip":
                wrap(
                    server.attack,
                    "poison_labels",
                    after=lambda result, seconds, labels, *a: event(
                        "attack_executed",
                        client=state["client"],
                        kind=kind,
                        changed_labels=int((result != labels).sum()),
                    ),
                )
            if kind == "backdoor":
                wrap(
                    server.attack,
                    "poison_data",
                    after=lambda result, seconds, cid, data: event(
                        "attack_executed",
                        kind=kind,
                        client=cid,
                        changed_labels=int((result.y_train != data.y_train).sum()),
                    ),
                )
            if hasattr(server.attack, "poison_batch"):
                wrap(
                    server.attack,
                    "poison_batch",
                    after=lambda result, seconds, *a: event(
                        "attack_executed", kind=kind, clients=sorted(server.attackers), duration_s=seconds
                    ),
                )

    def step_completed(result, seconds, optimizer, *args, **kwargs):
        if state["training"]:
            state["steps"] += 1
            if state["steps"] == 1:
                event(
                    "optimizer_step",
                    client=state["client"],
                    duration_s=seconds,
                    learning_rates=[g["lr"] for g in optimizer.param_groups],
                    gradient_norms=[
                        float(p.grad.detach().norm())
                        for g in optimizer.param_groups
                        for p in g["params"]
                        if p.grad is not None
                    ][:12],
                    scope="first step; first twelve parameter gradient norms",
                )

    def masking_completed(result, seconds, tensor, mask, value):
        if (
            state["training"]
            and state["steps"] == 0
            and value == -1e9
            and tensor.ndim == 2
            and tensor.shape[1] == state.get("num_classes")
        ):
            head = ("parent", "local")[min(state["mask_calls"], 1)]
            state["mask_calls"] += 1
            event(
                "aclm_applied",
                client=state["client"],
                head=head,
                active_mask=(~mask[0]).tolist(),
                masked_value=value,
                before=tensor[0].detach().tolist(),
                after=result[0].detach().tolist(),
                scope="first sample of first training batch",
                duration_s=seconds,
            )

    wrap(torch.Tensor, "masked_fill", after=masking_completed)
    from rfl.defense import soft_cosine

    def bounded(result, seconds, stack, k):
        if state.get("aggregating"):
            event(
                "update_norms_bounded",
                **state["aggregation_context"],
                norm_bound=k,
                before=stack.norm(dim=1).tolist(),
                after=result.norm(dim=1).tolist(),
                duration_s=seconds,
            )

    def cosines(result, seconds, *args, **kwargs):
        if state.get("aggregating"):
            state["cosines"] = result.tolist()

    def consensus(result, seconds, updates, temperature, hard_rejection, hard_threshold, class_masks, classifier):
        if state.get("aggregating"):
            event(
                "sccf_computed",
                **state["aggregation_context"],
                cosines=state["cosines"],
                trust=result.tolist(),
                base_temperature=temperature,
                hard_rejection=hard_rejection,
                hard_threshold=hard_threshold,
                uses_class_mask=class_masks is not None,
                duration_s=seconds,
            )

    wrap(soft_cosine, "bound_update_norms", after=bounded)
    wrap(soft_cosine.F, "cosine_similarity", after=cosines)
    wrap(soft_cosine, "soft_cosine_trust", after=consensus)

    def backward_completed(result, seconds, tensor, *args, **kwargs):
        if state["training"] and tensor.numel() == 1:
            state["last_loss"] = float(tensor.detach())
            if state["steps"] == 0:
                event(
                    "loss_backward_completed",
                    client=state["client"],
                    objective=state["last_loss"],
                    duration_s=seconds,
                    scope="first scalar backward objective",
                )

    wrap(torch.Tensor, "backward", after=backward_completed)
    wrap(torch.optim.SGD, "step", after=step_completed)
    saved = dict(registry.METHODS)

    def builder(build):
        def call(cfg, model, clients):
            server = build(cfg, model, clients)
            instrument(server, cfg, model, clients)
            return server

        return call

    try:
        for name, build in saved.items():
            registry.METHODS[name] = builder(build)
        wrap(
            runner,
            "evaluate_clients",
            after=lambda result, seconds, *a, **k: event(
                "evaluation_completed", values=asdict(result), duration_s=seconds
            ),
        )
        from rfl.server import fedhep_plus_server

        wrap(
            fedhep_plus_server,
            "screening_scores",
            after=lambda result, seconds, *a, **k: event(
                "similarity_computed", scores=result.tolist(), duration_s=seconds
            ),
        )
        yield
    finally:
        for restore in reversed(undo):
            restore()
        registry.METHODS.clear()
        registry.METHODS.update(saved)
