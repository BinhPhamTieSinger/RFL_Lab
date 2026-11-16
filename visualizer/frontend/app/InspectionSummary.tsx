type RecordValues = Record<string, any>;

function format(value: unknown): string {
  if (typeof value === "number")
    return Number.isInteger(value)
      ? String(value)
      : Number(value.toPrecision(4)).toString();
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (Array.isArray(value))
    return value.length ? value.map(format).join(", ") : "None";
  return String(value);
}

export function Facts({ values }: { values: RecordValues }) {
  const rows = Object.entries(values).filter(
    ([, v]) => v !== undefined && v !== null,
  );
  if (!rows.length) return <p>No values recorded at this step.</p>;
  return (
    <dl className="inspection-facts">
      {rows.map(([label, value]) => (
        <div key={label}>
          <dt>{label}</dt>
          <dd>{format(value)}</dd>
        </div>
      ))}
    </dl>
  );
}

export function RoundSummary({ data }: { data: RecordValues }) {
  if (!Object.keys(data).length)
    return <p>No diagnostics recorded at this step.</p>;
  const distribution = data.score_stats ?? {};
  const ids = Object.keys(data.scores ?? {});
  return (
    <>
      <Facts
        values={{
          Round: data.round,
          "Aggregation rule": data.rule,
          "Accepted clients": data.accepted,
          "Banned clients": data.banned,
          Regrouping: data.regroup,
          "Member agreement": data.member_agreement,
          "Parents copied global model": data.parents_copy_global,
          "Isolated clients": data.isolated,
          "Excluded this round": data.excluded,
          "Median similarity": distribution.median,
          "Median absolute deviation": distribution.mad,
          "Screening scale": distribution.scale,
          "Rewind target round": data.rewind_target,
        }}
      />
      {ids.length > 0 && (
        <details>
          <summary>Client scores ({ids.length})</summary>
          <table>
            <thead>
              <tr>
                <th>Client</th>
                <th>{data.score_stats ? "Similarity" : "Score"}</th>
                <th>z-score</th>
              </tr>
            </thead>
            <tbody>
              {ids.map((id) => (
                <tr key={id}>
                  <th>{id}</th>
                  <td>{format(data.scores[id])}</td>
                  <td>{format(data.z?.[id] ?? "—")}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </details>
      )}
      <RawRecord value={data} />
    </>
  );
}

export function RawRecord({ value }: { value: RecordValues }) {
  return (
    <details>
      <summary>Full recorded values</summary>
      <pre>{JSON.stringify(value, null, 2)}</pre>
    </details>
  );
}

export function EventSummary({ event }: { event?: RecordValues }) {
  if (!event) return <p>No event recorded at this step.</p>;
  return (
    <>
      <Facts
        values={{
          Operation: event.type?.replaceAll("_", " "),
          Round: event.round,
          Client: event.client,
          Parent: event.parent,
          Tier: event.tier,
          Rule: event.rule,
          Module: event.module,
          "Duration (s)": event.duration_s,
          "Optimizer steps": event.optimizer_steps,
          "Last training objective": event.last_backward_objective,
          "Update norm": event.output_norm ?? event.update_norm,
          "Contributing clients": event.members,
          "Training examples": event.samples,
        }}
      />
      <RawRecord value={event} />
    </>
  );
}
