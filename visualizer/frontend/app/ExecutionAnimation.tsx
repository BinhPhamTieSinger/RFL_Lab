"use client";
import { BaseEdge, getBezierPath, type EdgeProps } from "@xyflow/react";
import { useReducedMotion, motion } from "framer-motion";
import { Formula } from "./ResearchViews";

export function WeightEdge(props: EdgeProps) {
  const [path, middleX, middleY] = getBezierPath(props);
  const reduced = useReducedMotion();
  const active = Boolean(props.data?.active);
  return (
    <>
      <BaseEdge
        path={path}
        markerEnd={props.markerEnd}
        style={{ ...props.style, strokeWidth: active ? 3 : 1 }}
      />
      {active && (
        <circle
          cx={reduced || !props.data?.playing ? middleX : 0}
          cy={reduced || !props.data?.playing ? middleY : 0}
          r="5"
          fill="var(--text)"
          className="weight-packet"
          aria-label="Recorded weight transfer"
        >
          {reduced || !props.data?.playing ? (
            <animateMotion path={path} dur="1s" begin="indefinite" />
          ) : (
            <animateMotion
              key={String(props.data?.sequence)}
              path={path}
              dur="1s"
              repeatCount="indefinite"
            />
          )}
        </circle>
      )}
    </>
  );
}
export const edgeTypes = { weight: WeightEdge };
export function eventPhase(event: any) {
  if (event?.type === "evaluation_completed") return "Evaluation";
  if (
    ["similarity_computed", "screening_updated", "client_isolated"].includes(
      event?.type,
    )
  )
    return "Screening";
  if (
    [
      "aggregation_started",
      "aggregation_completed",
      "update_norms_bounded",
      "sccf_computed",
      "temporal_trust_updated",
    ].includes(event?.type)
  )
    return event.tier === "parent" || event.tier === "parent_defense"
      ? "Parents"
      : "Global";
  if (event?.type === "model_updated") return "Global";
  return "Clients";
}
export function eventFocus(event: any) {
  if (event?.type === "upload_ready")
    return `c${event.client}|${event.tier === "parent" ? (event.parent === undefined ? `c${event.client}` : `p${event.parent}`) : "global"}`;
  if (eventPhase(event) === "Parents") return `p${event.parent}`;
  if (typeof event?.client === "number") return `c${event.client}`;
  return "global";
}
export function isPlaybackStep(event: any) {
  return ![
    "log",
    "experiment_started",
    "experiment_pending",
    "experiment_queued",
    "worker_stopped",
    "experiment_completed",
  ].includes(event.type);
}
const names: Record<string, string> = {
  initialized: "Configured hierarchy and actual client partitions",
  client_training_started: "Client receives model state and starts training",
  operation_completed: "ResNet-9 computation",
  aclm_applied: "ACLM · active-class logit masking",
  loss_backward_completed: "Loss and backpropagation",
  optimizer_step: "SGD weight update",
  ensemble_mixing_updated: "Three-head ensemble mixing",
  client_training_completed: "Local model training completed",
  upload_ready: "Weight update ready for transfer",
  aggregation_started: "Server receives client weight updates",
  update_norms_bounded: "SCCF · bound update norms",
  sccf_computed: "SCCF · cosine similarity and trust weights",
  temporal_trust_updated: "TTT · temporal trust and isolation",
  screening_updated: "FedHEP+ · screening evidence and isolation",
  similarity_computed: "Global screening similarities",
  aggregation_completed: "Server applies the aggregated update",
  model_updated: "Global model updated",
  evaluation_completed: "Model evaluation",
};
const value = (n: any) =>
  typeof n === "number" ? n.toPrecision(5) : n === undefined ? "—" : String(n);
export function ExecutionAnimation({
  event,
  sequence,
  initialized,
  formulas,
}: any) {
  if (!event || !initialized) return null;
  const tier =
    event.tier === "parent"
      ? event.parent === undefined
        ? "parent upload buffer"
        : `Parent ${event.parent}`
      : "Global";
  const formulaNames =
    event.type === "aclm_applied"
      ? ["ACLM · active-class logit masking"]
      : event.type === "temporal_trust_updated"
        ? ["TTT · temporal trust tracking"]
        : [
              "sccf_computed",
              "update_norms_bounded",
              "aggregation_completed",
            ].includes(event.type)
          ? ["SCCF aggregation"]
          : event.type === "screening_updated"
            ? ["Temporal evidence", "Screening scale"]
            : event.type === "loss_backward_completed"
              ? ["Three-head objective"]
              : [];
  const members = event.members ?? [];
  const logitIndices =
    event.type === "aclm_applied"
      ? event.active_mask
          .map((_: boolean, i: number) => i)
          .sort(
            (a: number, b: number) =>
              Number(event.active_mask[a]) - Number(event.active_mask[b]),
          )
          .slice(0, 10)
      : [];
  return (
    <section
      className="execution-animation"
      aria-label="Animated operation inspection"
    >
      <div className="section-actions">
        <h2>{names[event.type] ?? event.type.replaceAll("_", " ")}</h2>
        <span>
          Round {event.round ?? 0} · captured at{" "}
          {event.elapsed_s?.toFixed(3) ?? "—"} s
        </span>
      </div>
      <motion.div
        key={sequence}
        initial={{ opacity: 0.7 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.15 }}
      >
        {event.client !== undefined && (
          <p>
            Client C_{event.client}
            {initialized.attackers.includes(event.client)
              ? " · configured attacker"
              : " · honest client"}
          </p>
        )}
        {event.type === "upload_ready" && (
          <p>
            C_{event.client} → {tier}. {event.bytes?.toLocaleString()} bytes ·
            update norm {value(event.update_norm)}. The animation represents the
            recorded in-process tensor upload.
          </p>
        )}
        {event.type === "operation_completed" && (
          <p>
            {event.operation} through {event.module} · {event.module_type}{" "}
            {event.shape?.join(" × ") ??
              `gradient norm ${value(event.gradient_norm)}`}
          </p>
        )}
        {event.type === "aclm_applied" && (
          <>
            <p>
              {event.head} head · {event.active_mask.filter(Boolean).length}/
              {event.active_mask.length} active classes. Root logits remain
              unmasked.
            </p>
            <div className="class-mask" aria-label="Actual active-class mask">
              {event.active_mask.map((active: boolean, i: number) => (
                <span
                  key={i}
                  className={active ? "present" : "absent"}
                  title={`Class ${i} · ${active ? "active" : "masked to −1e9"}`}
                >
                  {i}
                </span>
              ))}
            </div>
            <div className="table-scroll">
              <table>
                <thead>
                  <tr>
                    <th>Class</th>
                    <th>Before mask</th>
                    <th>After mask</th>
                  </tr>
                </thead>
                <tbody>
                  {logitIndices.map((i: number) => (
                    <tr key={i}>
                      <td>{i}</td>
                      <td>{value(event.before[i])}</td>
                      <td>{value(event.after[i])}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <small>
              Sample logits from the captured first sample, showing masked
              classes first. The class mask shows every class.
            </small>
          </>
        )}
        {event.type === "sccf_computed" && (
          <>
            <p>
              {tier} · class-mask calibration{" "}
              {event.uses_class_mask ? "enabled" : "disabled"} · base
              temperature {event.base_temperature} · hard rejection{" "}
              {event.hard_rejection
                ? `cosine < ${event.hard_threshold}`
                : "disabled"}
            </p>
            <table>
              <thead>
                <tr>
                  <th>Client</th>
                  <th>Cosine</th>
                  <th>Trust weight</th>
                  <th>Outcome</th>
                </tr>
              </thead>
              <tbody>
                {event.trust.map((t: number, i: number) => (
                  <tr key={i}>
                    <td>C_{members[i] ?? i}</td>
                    <td>{value(event.cosines?.[i])}</td>
                    <td>
                      <div className="trust-meter">
                        <span style={{ width: `${100 * t}%` }} />
                      </div>
                      {value(t)}
                    </td>
                    <td>
                      {t === 0 ? "Rejected by SCCF" : "Weighted contribution"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </>
        )}
        {event.type === "update_norms_bounded" && (
          <table>
            <thead>
              <tr>
                <th>Client</th>
                <th>Input norm</th>
                <th>Bounded norm</th>
              </tr>
            </thead>
            <tbody>
              {event.before.map((n: number, i: number) => (
                <tr key={i}>
                  <td>C_{members[i] ?? i}</td>
                  <td>{value(n)}</td>
                  <td>{value(event.after[i])}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        {event.type === "temporal_trust_updated" && (
          <>
            <p>
              {tier} ledger · momentum {event.momentum} · isolate below{" "}
              {event.threshold}
            </p>
            <table>
              <thead>
                <tr>
                  <th>Client</th>
                  <th>Before</th>
                  <th>After</th>
                  <th>Decision</th>
                </tr>
              </thead>
              <tbody>
                {members.map((cid: number) => (
                  <tr key={cid}>
                    <td>C_{cid}</td>
                    <td>{value(event.before[cid])}</td>
                    <td>{value(event.after[cid])}</td>
                    <td>
                      {event.isolated.includes(cid) ? "Isolated" : "Retained"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </>
        )}
        {event.type === "screening_updated" && (
          <>
            <p>
              Global evidence update · isolate below −{event.evidence_cut}.
              Persistence and configured gates also apply.
            </p>
            <table>
              <thead>
                <tr>
                  <th>Client</th>
                  <th>z-score</th>
                  <th>Evidence</th>
                  <th>Isolation</th>
                </tr>
              </thead>
              <tbody>
                {Object.keys(event.z).map((cid) => (
                  <tr key={cid}>
                    <td>C_{cid}</td>
                    <td>{value(event.z[cid])}</td>
                    <td>{value(event.evidence[cid])}</td>
                    <td>
                      {event.isolated.includes(Number(cid))
                        ? "Isolated"
                        : "Retained"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </>
        )}
        {event.type === "aggregation_completed" && (
          <p>
            {tier} · {event.algorithm ?? event.rule} ·{" "}
            {event.members?.length ?? event.trust?.length ?? "captured"}{" "}
            incoming updates · resulting update norm {value(event.output_norm)}{" "}
            · {event.duration_s?.toFixed(4)} s
          </p>
        )}
        {formulas
          .filter(
            (f: any) =>
              formulaNames.includes(f.name) &&
              (f.name !== "SCCF aggregation" ||
                initialized.method !== "fedhep"),
          )
          .map((f: any) => (
            <div key={f.name}>
              <Formula>{`$$\n${f.latex}\n$$\n\n${f.explanation}`}</Formula>
              <code>{f.source}</code>
            </div>
          ))}
        <details>
          <summary>Actual event values</summary>
          <pre>{JSON.stringify(event, null, 2)}</pre>
        </details>
      </motion.div>
    </section>
  );
}
