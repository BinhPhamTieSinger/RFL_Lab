"use client";
import { useEffect, useState } from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
} from "recharts";
import Markdown from "react-markdown";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

type RecordValue = Record<string, any>;
export function Formula({ children }: { children: string }) {
  return (
    <Markdown remarkPlugins={[remarkMath]} rehypePlugins={[rehypeKatex]}>
      {children}
    </Markdown>
  );
}
export function Series({
  rows,
  keys,
  title,
  domain,
}: {
  rows: RecordValue[];
  keys: string[];
  title: string;
  domain?: [number, number];
}) {
  if (!rows.length) return null;
  return (
    <section className="research-chart">
      <h3>{title}</h3>
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={rows}>
          <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
          <XAxis dataKey="round" stroke="var(--muted)" />
          <YAxis stroke="var(--muted)" domain={domain} />
          <Tooltip
            contentStyle={{
              background: "var(--surface)",
              color: "var(--text)",
              borderColor: "var(--border)",
            }}
          />
          <Legend />
          {keys.map((key, i) => (
            <Line
              key={key}
              dataKey={key}
              type="linear"
              stroke={i % 2 ? "#888" : "var(--text)"}
              strokeDasharray={i ? `${i * 3} 3` : undefined}
              dot={rows.length < 10}
              connectNulls={false}
              isAnimationActive={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </section>
  );
}
export function ClientDistribution({
  client,
  malicious,
  parent,
}: {
  client: RecordValue;
  malicious: boolean;
  parent?: number;
}) {
  const top = client.histogram
    .map((n: number, label: number) => ({ n, label }))
    .filter((x: any) => x.n > 0)
    .sort((a: any, b: any) => b.n - a.n)
    .slice(0, 3);
  return (
    <section
      className={`client-distribution ${malicious ? "malicious" : "honest"}`}
    >
      <div className="client-heading">
        <h3>Client C_{client.id}</h3>
        {parent !== undefined && <span>P_{parent}</span>}
        <strong>{malicious ? "Malicious" : "Honest"}</strong>
      </div>
      <div
        className="histogram"
        role="img"
        aria-label={`Client ${client.id} class distribution`}
      >
        {client.histogram.map((n: number, i: number) => (
          <div
            key={i}
            title={`${client.class_names?.[i] ?? i} · ${n}`}
            style={{
              height: `${(80 * n) / Math.max(1, ...client.histogram)}px`,
            }}
          />
        ))}
      </div>
      <dl className="partition-stats">
        <div>
          <dt>Samples</dt>
          <dd>{client.train}</dd>
        </div>
        <div>
          <dt>Classes</dt>
          <dd>
            {client.classes_present ??
              client.histogram.filter((n: number) => n > 0).length}
            /{client.histogram.length}
          </dd>
        </div>
        <div>
          <dt>R_skew</dt>
          <dd>{client.r_skew?.toFixed(3) ?? "Uncaptured"}</dd>
        </div>
      </dl>
      <p>
        {top
          .map(
            (x: any) => `${client.class_names?.[x.label] ?? x.label} · ${x.n}`,
          )
          .join(" / ")}
      </p>
      <div className="sample-images">
        {client.samples?.map((sample: any) => (
          <figure key={sample.partition_index}>
            <img
              src={sample.image}
              width={72}
              height={72}
              alt={`${sample.name}, client ${client.id}, partition index ${sample.partition_index}`}
            />
            <figcaption>{sample.name}</figcaption>
          </figure>
        ))}
      </div>
      <small>
        {client.sample_scope ??
          "Sample images were not captured in this earlier run."}
      </small>
    </section>
  );
}
function Values({ value }: { value: any }) {
  return <pre>{JSON.stringify(value, null, 2)}</pre>;
}
export function RoundWalkthrough({
  events,
  currentEvent,
  jump,
  initialized,
  formulas,
  focus,
}: {
  events: RecordValue[];
  currentEvent?: RecordValue;
  jump: (phase: string) => void;
  initialized?: RecordValue;
  formulas: RecordValue[];
  focus: (id: string) => void;
}) {
  const [chosenRound, setRound] = useState(1);
  const [phase, setPhase] = useState("Clients");
  const [honestId, setHonest] = useState<number | null>(null);
  const [maliciousId, setMalicious] = useState<number | null>(null);
  useEffect(() => {
    if (!currentEvent) return;
    if (currentEvent.round) setRound(currentEvent.round);
    const tier = currentEvent.tier;
    if (currentEvent.type === "evaluation_completed") setPhase("Evaluation");
    else if (
      ["similarity_computed", "screening_updated", "client_isolated"].includes(
        currentEvent.type,
      )
    )
      setPhase("Screening");
    else if (
      [
        "aggregation_started",
        "aggregation_completed",
        "update_norms_bounded",
        "sccf_computed",
        "temporal_trust_updated",
      ].includes(currentEvent.type)
    )
      setPhase(
        tier === "parent" || tier === "parent_defense" ? "Parents" : "Global",
      );
    else if (currentEvent.type === "client_training_started")
      setPhase("Clients");
  }, [currentEvent]);
  if (!initialized) return null;
  const rounds = Array.from(
    new Set(
      events.filter((e) => e.type === "round_started").map((e) => e.round),
    ),
  );
  const records = events.filter((e) => e.round === chosenRound);
  const diagnostics = records.find((e) => e.type === "diagnostics")?.values;
  const training = records.filter(
    (e) => e.type === "client_training_completed",
  );
  const attackers = initialized.attackers;
  const honest = initialized.clients.filter(
    (c: any) => !attackers.includes(c.id),
  );
  const malicious = initialized.clients.filter((c: any) =>
    attackers.includes(c.id),
  );
  const hierarchy = initialized.method.startsWith("fedhep");
  const phases = [
    "Clients",
    ...(initialized.method === "fedhep_plus" ? ["Screening"] : []),
    ...(initialized.method === "fedhep_robust"
      ? ["Global", "Parents"]
      : hierarchy
        ? ["Parents", "Global"]
        : ["Global"]),
    "Evaluation",
  ];
  const relevant =
    phase === "Screening"
      ? records.filter((e) =>
          [
            "similarity_computed",
            "screening_updated",
            "client_isolated",
            "diagnostics",
            "detection",
          ].includes(e.type),
        )
      : phase === "Evaluation"
        ? records.filter((e) => e.type === "evaluation_completed")
        : records.filter(
            (e) =>
              [
                "aggregation_started",
                "aggregation_completed",
                "update_norms_bounded",
                "sccf_computed",
                "temporal_trust_updated",
              ].includes(e.type) &&
              (phase === "Parents"
                ? ["parent", "parent_defense"].includes(e.tier)
                : !["parent", "parent_defense"].includes(e.tier)),
          );
  const formulaIds =
    phase === "Clients"
      ? [
          "ACLM · active-class logit masking",
          "Three-head objective",
          "Label distribution",
          "Ensemble mixing update",
          "Learning rate",
        ]
      : phase === "Screening"
        ? [
            "Screening scale",
            "Two-sided screen and gap gate",
            "Critical-layer and sign-alignment signals",
            "Temporal evidence",
            "Rewind on isolation",
          ]
        : [
            "Sample-weighted mean",
            "SCCF aggregation",
            "L0 · influence bound (ARC, NNM, trimmed mean)",
            "TTT · temporal trust tracking",
          ];
  const selectedFormulas = formulas.filter(
    (f) =>
      formulaIds.includes(f.name) &&
      (f.method === "all" || initialized.method.startsWith(f.method)) &&
      (f.name !== "SCCF aggregation" || initialized.method !== "fedhep"),
  );
  return (
    <section className="round-walkthrough">
      <div className="section-actions">
        <h2>Inside a round</h2>
        <label>
          Round{" "}
          <select
            aria-label="Inspect round"
            value={chosenRound}
            onChange={(e) => setRound(Number(e.target.value))}
          >
            {(rounds.length ? rounds : [1]).map((r) => (
              <option key={r} value={r}>
                {r}
              </option>
            ))}
          </select>
        </label>
      </div>
      <p>
        {training.length}/{initialized.clients.length} client training calls
        completed in this round.{" "}
        {initialized.attack?.kind
          ? `Configured attack · ${initialized.attack.kind}`
          : "Attack disabled"}
      </p>
      <nav className="phase-nav" aria-label="Round phases">
        {phases.map((name) => (
          <button
            key={name}
            aria-pressed={phase === name}
            onClick={() => {
              setPhase(name);
              jump(name);
              focus(
                name === "Clients"
                  ? `c${honestId ?? honest[0]?.id ?? 0}`
                  : name === "Parents"
                    ? `p${initialized.parents[0]}`
                    : "global",
              );
            }}
          >
            {name}
          </button>
        ))}
      </nav>
      {phase === "Clients" ? (
        <div className="client-comparison">
          {[
            {
              label: "Honest",
              clients: honest,
              selected: honestId,
              change: setHonest,
            },
            {
              label: "Malicious",
              clients: malicious,
              selected: maliciousId,
              change: setMalicious,
            },
          ].map((side) => {
            const client =
              side.clients.find((c: any) => c.id === side.selected) ??
              side.clients[0];
            if (!client)
              return (
                <section key={side.label}>
                  <h3>{side.label} clients</h3>
                  <p>This run has no {side.label.toLowerCase()} clients.</p>
                </section>
              );
            const captured = records.filter(
              (e) => e.client === client.id || e.clients?.includes(client.id),
            );
            const done = captured.find(
              (e) => e.type === "client_training_completed",
            );
            return (
              <section key={side.label}>
                <label>
                  {side.label} client{" "}
                  <select
                    value={client.id}
                    onChange={(e) => {
                      side.change(Number(e.target.value));
                      focus(`c${e.target.value}`);
                    }}
                  >
                    {side.clients.map((c: any) => (
                      <option key={c.id} value={c.id}>
                        C_{c.id}
                      </option>
                    ))}
                  </select>
                </label>
                <ClientDistribution
                  client={client}
                  malicious={side.label === "Malicious"}
                  parent={
                    (diagnostics?.groups ?? initialized.groups)[client.id]
                  }
                />
                <p>
                  {done
                    ? `${done.optimizer_steps} SGD steps · ${done.duration_s.toFixed(3)} s · last objective ${done.last_backward_objective?.toFixed(5) ?? "uncaptured"}`
                    : "Waiting for the training completion event."}
                </p>
                {captured
                  .filter((e) =>
                    [
                      "attack_executed",
                      "upload_ready",
                      "loss_backward_completed",
                      "optimizer_step",
                      "ensemble_mixing_updated",
                      "aclm_applied",
                    ].includes(e.type),
                  )
                  .map((e, i) => (
                    <details key={i}>
                      <summary>
                        {e.type.replaceAll("_", " ")}
                        {e.tier ? ` · ${e.tier}` : ""}
                      </summary>
                      <Values value={e} />
                    </details>
                  ))}
                <details>
                  <summary>Forward and backward operations</summary>
                  <table>
                    <thead>
                      <tr>
                        <th>Operation</th>
                        <th>Module</th>
                        <th>Captured value</th>
                      </tr>
                    </thead>
                    <tbody>
                      {captured
                        .filter((e) => e.type === "operation_completed")
                        .map((e, i) => (
                          <tr key={i}>
                            <td>{e.operation}</td>
                            <td>{e.module}</td>
                            <td>
                              {e.shape?.join(" × ") ??
                                e.gradient_norm?.toPrecision(5)}
                            </td>
                          </tr>
                        ))}
                    </tbody>
                  </table>
                </details>
              </section>
            );
          })}
        </div>
      ) : (
        <div>
          <p>
            {phase === "Screening"
              ? "FedHEP+ Global screening updates evidence and isolation before parent SCCF aggregation."
              : phase === "Parents"
                ? "Recorded parent aggregation inputs and trust values."
                : phase === "Evaluation"
                  ? "Evaluation uses the research runner’s actual client population."
                  : "Recorded Global aggregation and resulting model updates."}
          </p>
          {relevant.length ? (
            relevant.map((e, i) => (
              <details key={i}>
                <summary>
                  {e.type.replaceAll("_", " ")}{" "}
                  {e.parent !== undefined ? `· parent ${e.parent}` : ""}
                  {e.duration_s !== undefined
                    ? ` · ${e.duration_s.toFixed(4)} s`
                    : ""}
                </summary>
                <Values value={e} />
              </details>
            ))
          ) : (
            <p>No event for this phase has arrived in this round.</p>
          )}
        </div>
      )}
      {selectedFormulas.map((f) => (
        <details key={f.name}>
          <summary>{f.name} · implementation formula</summary>
          <Formula>{`$$
${f.latex}
$$

${f.explanation}`}</Formula>
          <code>{f.source}</code>
        </details>
      ))}
    </section>
  );
}

const resultCache = new Map<string, Promise<any>>();
function fetchResult(id: string, refresh = false) {
  if (refresh) resultCache.delete(id);
  if (!resultCache.has(id))
    resultCache.set(
      id,
      fetch(`/research/${id}.json`, { cache: refresh ? "reload" : "default" })
        .then((response) => {
          if (!response.ok) throw Error("Result could not be loaded");
          return response.json();
        })
        .catch((error) => {
          resultCache.delete(id);
          throw error;
        }),
    );
  return resultCache.get(id)!;
}
function RawResult({ result }: any) {
  const [open, setOpen] = useState(false);
  return (
    <details onToggle={(e) => setOpen(e.currentTarget.open)}>
      <summary>Full saved result and diagnostics</summary>
      {open && <Values value={result} />}
    </details>
  );
}
function ResultDetails({
  run,
  loadConfig,
  replay,
  cancel,
  expanded = false,
  checked = false,
  select,
}: any) {
  const [detail, setDetail] = useState<any>(null);
  const [opened, setOpened] = useState(expanded);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [copied, setCopied] = useState(false);
  async function load(refresh = false) {
    setLoading(true);
    setError("");
    try {
      setDetail(await fetchResult(run.id, refresh));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }
  const result = detail?.result;
  const mad = (result?.diagnostics ?? [])
    .filter((d: any) => d.score_stats)
    .map((d: any) => ({ round: d.round, ...d.score_stats }));
  return (
    <details
      className="experiment-entry"
      id={run.id}
      open={expanded || undefined}
      onToggle={(e) => {
        setOpened(e.currentTarget.open);
        if (e.currentTarget.open && !detail && !loading && !error) load();
      }}
    >
      <summary>
        <label
          className="selection-control"
          onClick={(e) => e.stopPropagation()}
        >
          <input
            type="checkbox"
            aria-label={`Select ${run.name ?? run.id}`}
            checked={checked}
            onChange={() => select?.(run.id)}
          />
        </label>
        <strong>{run.name ?? run.config.method}</strong>
        <span>{run.status === "running" ? "in progress" : run.status}</span>
        <span>
          {run.archived
            ? run.source
            : new Date(run.created * 1000).toLocaleString()}
        </span>
      </summary>
      {opened && (
        <div className="experiment-content">
          {run.notes && <Formula>{run.notes}</Formula>}
          <div className="section-actions">
            <button onClick={() => loadConfig(run.config)}>
              Load configuration · CPU
            </button>
            {!run.archived && (
              <button onClick={() => replay(run)}>Replay events</button>
            )}
            {["pending", "queued", "running"].includes(run.status) && (
              <button onClick={() => cancel(run.id)}>Cancel</button>
            )}
            <button onClick={() => load(true)}>Reload results</button>
            <a
              href={`?experiment=${encodeURIComponent(run.id)}&section=config`}
            >
              Configuration link
            </a>
            <button
              onClick={async () => {
                const url = new URL(
                  window.location.pathname,
                  window.location.origin,
                );
                url.searchParams.set("experiment", run.id);
                url.searchParams.set("section", "config");
                try {
                  await navigator.clipboard.writeText(url.toString());
                  setCopied(true);
                } catch {
                  setError(
                    "Use the Configuration link to copy this experiment’s address.",
                  );
                }
              }}
            >
              {copied ? "Link copied" : "Copy configuration link"}
            </button>
          </div>
          {loading && <p>Loading saved result…</p>}
          {error && <p role="alert">{error}</p>}
          <details open={expanded || undefined}>
            <summary>Configuration</summary>
            <table>
              <tbody>
                {Object.entries(run.config.overrides).map(([key, value]) => (
                  <tr key={key}>
                    <th>{key}</th>
                    <td>{String(value)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </details>
          {result ? (
            <>
              <Series
                title="Evaluation accuracy (%)"
                rows={result.history ?? []}
                keys={["pooled_acc", "mean_client_acc", "bottom10"]}
              />
              {result.history?.some((r: any) => r.asr != null) && (
                <Series
                  title="Backdoor success rate (%)"
                  rows={result.history}
                  keys={["asr"]}
                />
              )}
              {result.history?.some((r: any) => r.head_agreement || r.guarded) && (
                <>
                  <Series
                    title="Prediction-time defenses: accuracy (%)"
                    rows={predictionRows(result.history)}
                    keys={["model", "head_agreement", "exit_guard"]}
                  />
                  {result.history.some((r: any) => r.asr != null) && (
                    <Series
                      title="Prediction-time defenses: backdoor success (%)"
                      rows={predictionRows(result.history, "asr")}
                      keys={["model", "head_agreement", "exit_guard"]}
                    />
                  )}
                </>
              )}
              <Series
                title="MAD and screening scale per round"
                rows={mad}
                keys={["mad", "scale", "median"]}
              />
              <Series
                title="Detection rates (0–1)"
                rows={result.detection ?? []}
                domain={[0, 1]}
                keys={["precision", "recall", "false_isolation_rate"]}
              />
              <h3>Saved evaluation table</h3>
              <div className="table-scroll">
                <table>
                  <thead>
                    <tr>
                      <th>Round</th>
                      <th>Pooled accuracy %</th>
                      <th>Mean client %</th>
                      <th>Bottom 10% %</th>
                      <th>Backdoor success %</th>
                      <th>Elapsed s</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(result.history ?? []).map((r: any) => (
                      <tr key={r.round}>
                        <td>{r.round}</td>
                        <td>{r.pooled_acc?.toFixed(3)}</td>
                        <td>{r.mean_client_acc?.toFixed(3)}</td>
                        <td>{r.bottom10?.toFixed(3)}</td>
                        <td>{r.asr?.toFixed(2) ?? "–"}</td>
                        <td>{r.elapsed_s?.toFixed(2)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <RawResult result={result} />
            </>
          ) : (
            detail && <p>No final metrics file is available for this entry.</p>
          )}
          {detail?.logs?.map((log: any) => (
            <details key={log.name}>
              <summary>{log.name}</summary>
              <pre>{log.text}</pre>
            </details>
          ))}
          {run.archived && (
            <p>
              Source {run.source}. Date uses file modification time. Archived
              event replay requires captured visualizer events.
            </p>
          )}
        </div>
      )}
    </details>
  );
}
export function ExperimentHistory({
  targetId,
  history,
  refresh,
  loadConfig,
  replay,
  cancel,
}: any) {
  const [query, setQuery] = useState("");
  const [limit, setLimit] = useState(25);
  const [selected, setSelected] = useState<string[]>([]);
  const [onlySelected, setOnlySelected] = useState(false);
  const [shareLink, setShareLink] = useState("");
  useEffect(() => {
    const url = new URL(window.location.href);
    setQuery(url.searchParams.get("q") ?? "");
    let sharedIds: string | null = null;
    const encoded = url.searchParams.get("selection");
    if (encoded) {
      try {
        const hex = Array.from(
          atob(encoded.replace(/-/g, "+").replace(/_/g, "/")),
        )
          .map((byte) => byte.charCodeAt(0).toString(16).padStart(2, "0"))
          .join("");
        if (hex.length && hex.length % 24 === 0)
          sharedIds = hex
            .match(/.{24}/g)!
            .map((id) => `archive-${id}`)
            .join(",");
      } catch {
        /* Ignore malformed shared selections. */
      }
    }
    const ids = (
      sharedIds ??
      url.searchParams.get("experiments") ??
      localStorage.getItem("rfl-selected-experiments") ??
      ""
    )
      .split(",")
      .filter(Boolean);
    setSelected(ids);
    if (sharedIds !== null || url.searchParams.has("experiments")) {
      localStorage.setItem("rfl-selected-experiments", ids.join(","));
      url.searchParams.delete("experiments");
      url.searchParams.delete("selection");
      window.history.replaceState(null, "", url);
    }
    setOnlySelected(url.searchParams.get("onlySelected") === "1");
  }, []);
  function updateSelection(ids: string[], only: boolean) {
    setSelected(ids);
    setShareLink("");
    setOnlySelected(only);
    localStorage.setItem("rfl-selected-experiments", ids.join(","));
    const url = new URL(window.location.href);
    url.searchParams.delete("experiments");
    url.searchParams.delete("selection");
    if (only) {
      url.searchParams.set("onlySelected", "1");
      url.searchParams.delete("experiment");
    } else url.searchParams.delete("onlySelected");
    window.history.replaceState(null, "", url);
  }
  async function shareSelection() {
    const hex = selected.map((id) => id.replace(/^archive-/, "")).join("");
    const bytes = hex
      .match(/../g)!
      .map((pair) => String.fromCharCode(parseInt(pair, 16)))
      .join("");
    const encoded = btoa(bytes)
      .replace(/\+/g, "-")
      .replace(/\//g, "_")
      .replace(/=+$/, "");
    const url = new URL(window.location.origin + window.location.pathname);
    url.searchParams.set("tab", "Experiment history");
    url.searchParams.set("onlySelected", "1");
    url.searchParams.set("selection", encoded);
    setShareLink(url.href);
    try {
      await navigator.clipboard.writeText(url.href);
    } catch {
      /* The visible field supports manual copying. */
    }
  }
  function select(id: string) {
    updateSelection(
      selected.includes(id)
        ? selected.filter((value) => value !== id)
        : [...selected, id],
      onlySelected,
    );
  }

  const target = history.find((r: any) => r.id === targetId);
  const filtered = history.filter(
    (r: any) =>
      (r.id !== targetId || onlySelected) &&
      (!onlySelected || selected.includes(r.id)) &&
      `${r.name} ${r.config.method} ${r.status} ${r.source}`
        .toLowerCase()
        .includes(query.toLowerCase()),
  );
  return (
    <>
      <h1>Experiment history</h1>
      <p>
        {history.length} curated research entries. Edit
        research/history-manifest.json to add or remove experiments. Published
        changes refresh every 15 minutes.
      </p>
      <div className="section-actions">
        <input
          aria-label="Search experiments"
          placeholder="Search method, experiment or status"
          value={query}
          onChange={(e) => {
            setQuery(e.target.value);
            setLimit(25);
            const url = new URL(window.location.href);
            if (e.target.value) url.searchParams.set("q", e.target.value);
            else url.searchParams.delete("q");
            window.history.replaceState(null, "", url);
          }}
        />
        <button onClick={refresh}>Refresh history</button>
        <button
          disabled={!selected.length && !onlySelected}
          onClick={() => updateSelection(selected, !onlySelected)}
        >
          {onlySelected
            ? "Show all experiments"
            : `Show selected (${selected.length})`}
        </button>
        <button disabled={!selected.length} onClick={shareSelection}>
          Copy selected link
        </button>
        {selected.length > 0 && (
          <button onClick={() => updateSelection([], false)}>
            Clear selection
          </button>
        )}
      </div>
      {shareLink && (
        <label>
          Share these selected experiments
          <input
            aria-label="Selected experiments link"
            readOnly
            value={shareLink}
            onFocus={(event) => event.target.select()}
          />
        </label>
      )}
      {target && !onlySelected && (
        <ResultDetails
          key={target.id}
          run={target}
          expanded
          checked={selected.includes(target.id)}
          select={select}
          {...{ loadConfig, replay, cancel }}
        />
      )}
      {targetId && history.length > 0 && !target && (
        <p role="alert">
          This experiment is absent from the current curated history.
        </p>
      )}
      {filtered.slice(0, limit).map((run: any) => (
        <ResultDetails
          key={run.id}
          checked={selected.includes(run.id)}
          select={select}
          {...{ run, loadConfig, replay, cancel }}
        />
      ))}
      {filtered.length > limit && (
        <button onClick={() => setLimit(limit + 25)}>
          Show 25 more ({filtered.length - limit} remaining)
        </button>
      )}
    </>
  );
}

/** One row per evaluated round: the model's own number next to the same number under head-agreement inference
 * and the depth-exit guard (both evaluation-only, recorded by fedhep.head_agreement and fedhep.exit_guard). */
function predictionRows(history: any[], metric: "pooled_acc" | "asr" = "pooled_acc") {
  return history
    .filter((r) => r.head_agreement || r.guarded)
    .map((r) => ({
      round: r.round,
      model: r[metric],
      head_agreement: r.head_agreement?.[metric],
      exit_guard: r.guarded?.[metric],
    }));
}

