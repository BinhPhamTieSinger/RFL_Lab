"use client";
import { useState } from "react";
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
}: {
  rows: RecordValue[];
  keys: string[];
  title: string;
}) {
  if (!rows.length) return null;
  return (
    <section className="research-chart">
      <h3>{title}</h3>
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={rows}>
          <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
          <XAxis dataKey="round" stroke="var(--muted)" />
          <YAxis stroke="var(--muted)" />
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
  initialized,
  formulas,
  focus,
}: {
  events: RecordValue[];
  initialized?: RecordValue;
  formulas: RecordValue[];
  focus: (id: string) => void;
}) {
  const [chosenRound, setRound] = useState(1);
  const [phase, setPhase] = useState("Clients");
  const [honestId, setHonest] = useState<number | null>(null);
  const [maliciousId, setMalicious] = useState<number | null>(null);
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
            "client_isolated",
            "diagnostics",
            "detection",
          ].includes(e.type),
        )
      : phase === "Evaluation"
        ? records.filter((e) => e.type === "evaluation_completed")
        : records.filter(
            (e) =>
              e.type.startsWith("aggregation_") &&
              (phase === "Parents" ? e.tier === "parent" : e.tier !== "parent"),
          );
  const formulaIds =
    phase === "Clients"
      ? [
          "Three-head objective",
          "Label distribution",
          "Ensemble mixing update",
          "Learning rate",
        ]
      : phase === "Screening"
        ? ["Screening scale", "Temporal evidence"]
        : ["Sample-weighted mean", "SCCF aggregation"];
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
          <Formula>{`$$${f.latex}$$\n\n${f.explanation}`}</Formula>
          <code>{f.source}</code>
        </details>
      ))}
    </section>
  );
}

function ResultDetails({ run, loadConfig, replay, cancel }: any) {
  const [detail, setDetail] = useState<any>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  async function load() {
    setLoading(true);
    try {
      const response = await fetch(`/api/experiments/${run.id}`);
      if (!response.ok) throw Error("Result could not be loaded");
      setDetail(await response.json());
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
      onToggle={(e) => {
        if (e.currentTarget.open && !detail && !loading && !error) load();
      }}
    >
      <summary>
        <strong>{run.name ?? run.config.method}</strong>
        <span>{run.status === "running" ? "in progress" : run.status}</span>
        <span>
          {run.archived
            ? run.source
            : new Date(run.created * 1000).toLocaleString()}
        </span>
      </summary>
      <div className="experiment-content">
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
          <button onClick={load}>Reload results</button>
        </div>
        {loading && <p>Loading saved result…</p>}
        {error && <p role="alert">{error}</p>}
        <details>
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
            <Series
              title="MAD and screening scale per round"
              rows={mad}
              keys={["mad", "scale", "median"]}
            />
            <Series
              title="Detection rates (0–1)"
              rows={result.detection ?? []}
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
                      <td>{r.elapsed_s?.toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <details>
              <summary>Full saved result and diagnostics</summary>
              <Values value={result} />
            </details>
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
    </details>
  );
}
export function ExperimentHistory({
  history,
  refresh,
  loadConfig,
  replay,
  cancel,
}: any) {
  const [query, setQuery] = useState("");
  const [limit, setLimit] = useState(25);
  const filtered = history.filter((r: any) =>
    `${r.name} ${r.config.method} ${r.status} ${r.source}`
      .toLowerCase()
      .includes(query.toLowerCase()),
  );
  return (
    <>
      <h1>Experiment history</h1>
      <p>
        {history.length} saved entries. Refreshes every 15 minutes while this
        page is open. Live execution streams continuously.
      </p>
      <div className="section-actions">
        <input
          aria-label="Search experiments"
          placeholder="Search method, experiment or status"
          value={query}
          onChange={(e) => {
            setQuery(e.target.value);
            setLimit(25);
          }}
        />
        <button onClick={refresh}>Refresh history</button>
      </div>
      {filtered.slice(0, limit).map((run: any) => (
        <ResultDetails key={run.id} {...{ run, loadConfig, replay, cancel }} />
      ))}
      {filtered.length > limit && (
        <button onClick={() => setLimit(limit + 25)}>
          Show 25 more ({filtered.length - limit} remaining)
        </button>
      )}
    </>
  );
}
