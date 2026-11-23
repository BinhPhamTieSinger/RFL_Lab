"use client";
import { ReactNode, useEffect, useMemo, useRef, useState } from "react";
import DocAdvisor from "./DocAdvisor";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

type Document = { path: string; title: string; content: string };

type DocMeta = {
  path: string;
  shortTitle: string;
  subtitle: string;
  category: string;
};

type DecisionItem = {
  id: string;
  num: number;
  date: string;
  status: string;
  statusKind: "active" | "updated" | "closed";
  track: string;
  decision: string;
  why: string;
};

// Figure overlay zoom, relative to the whole figure fitted on screen (1 = fit page).
const ZOOM_MIN = 1;
const ZOOM_MAX = 8;

const DOC_CATALOG: DocMeta[] = [
  {
    path: "docs/report/00_index.md",
    shortTitle: "Research Report: Start Here",
    subtitle: "Research path, models at a glance, presenting kit",
    category: "0. Research Report",
  },
  {
    path: "docs/report/01_federated_learning.md",
    shortTitle: "1. Federated Learning & Attacks",
    subtitle: "FedAvg, non-IID data, settings D1/D2, attacks, metrics, notation",
    category: "0. Research Report",
  },
  {
    path: "docs/report/02_fedhep.md",
    shortTitle: "2. FedHEP (Phase 1)",
    subtitle: "Three-head model, formulas, changes vs FedAvg, reproduction",
    category: "0. Research Report",
  },
  {
    path: "docs/report/03_defended_fedhep.md",
    shortTitle: "3. Defended FedHEP (Phases 1c, 2)",
    subtitle: "SCCF and trust ledger, Phase 2 benchmark, failures",
    category: "0. Research Report",
  },
  {
    path: "docs/report/04_fedhep_plus.md",
    shortTitle: "4. FedHEP+ (Phase 3)",
    subtitle: "Robust z screen, results, follow-up diagnostics",
    category: "0. Research Report",
  },
  {
    path: "docs/report/05_ideas_that_did_not_work.md",
    shortTitle: "5. Ideas That Did Not Work",
    subtitle: "Drill-down, classic rules, dropped Phase 6 parts",
    category: "0. Research Report",
  },
  {
    path: "docs/report/06_fedhep_guard.md",
    shortTitle: "6. FedHEP-Guard: The Model",
    subtitle: "Every mechanism: formula, legend, location, configuration",
    category: "0. Research Report",
  },
  {
    path: "docs/report/07_fedhep_guard_results.md",
    shortTitle: "7. FedHEP-Guard: Results",
    subtitle: "Scorecard against FedHEP+, every experiment, all models",
    category: "0. Research Report",
  },
  {
    path: "docs/report/08_open_problems.md",
    shortTitle: "8. Open Problems",
    subtitle: "Honest isolations, their causes, Phase 8 experiments",
    category: "0. Research Report",
  },
  {
    path: "docs/report/09_benchmark.md",
    shortTitle: "9. Canonical Benchmark",
    subtitle: "Every model, same attacks, one machine",
    category: "0. Research Report",
  },
  {
    path: "docs/report/10_configuration_reference.md",
    shortTitle: "10. Configuration Reference",
    subtitle: "Every config key, symbol, preset and run prefix",
    category: "0. Research Report",
  },
  {
    path: "README.md",
    shortTitle: "RFL Lab Overview",
    subtitle: "Status, Phase 5–6 results, repo layout & reproducibility",
    category: "1. Start Here & Overview",
  },
  {
    path: "docs/RESEARCH_SUMMARY.md",
    shortTitle: "RFL Lab in Plain Words (earlier)",
    subtitle: "Walkthrough to Phase 6, superseded by the report",
    category: "1. Start Here & Overview",
  },
  {
    path: "docs/PAPER_PLAN.md",
    shortTitle: "Paper Plan",
    subtitle: "Claims C1–C9, related work & figure-to-run mapping",
    category: "1. Start Here & Overview",
  },
  {
    path: "docs/RESEARCH_PLAN.md",
    shortTitle: "FedHEP-Guard Plan",
    subtitle: "Layers L0–L3, hypotheses H1–H8 & ablation ladder",
    category: "2. Defense Design & Theory",
  },
  {
    path: "docs/PREDICTIONS.md",
    shortTitle: "Pre-Campaign Predictions",
    subtitle: "P1–P16 expected numbers & refutation criteria",
    category: "2. Defense Design & Theory",
  },
  {
    path: "docs/IDEAS.md",
    shortTitle: "Defense Ideas (1–19)",
    subtitle: "Ranked mechanisms from robust FL & neighbouring fields",
    category: "2. Defense Design & Theory",
  },
  {
    path: "docs/LITERATURE_REVIEW.md",
    shortTitle: "Literature Review",
    subtitle: "170 sources across aggregation, attacks & 7 open gaps",
    category: "2. Defense Design & Theory",
  },
  {
    path: "docs/EXPERIMENT_LOG.md",
    shortTitle: "Experiment Log",
    subtitle: "Complete laboratory record, charts & negative results",
    category: "3. Empirical & Decision Logs",
  },
  {
    path: "docs/DECISIONS.md",
    shortTitle: "Decision Log (D1–D33)",
    subtitle: "Architecture, parity, active defense & closed explorations",
    category: "3. Empirical & Decision Logs",
  },
  {
    path: "docs/ARCHITECTURE.md",
    shortTitle: "Package Architecture",
    subtitle: "5-layer dependency rule & plug-in slot interfaces",
    category: "4. Architecture & Operations",
  },
  {
    path: "docs/OPERATIONS.md",
    shortTitle: "Operations Guide",
    subtitle: "Mac (MPS), Acer (XPU) & Lab GPU Server (CUDA / Slurm)",
    category: "4. Architecture & Operations",
  },
  {
    path: "visualizer/README.md",
    shortTitle: "Visualizer Workbench",
    subtitle: "Local SSE runner, captured playback & deployment",
    category: "4. Architecture & Operations",
  },
];

function slugify(text: string): string {
  return text
    .toLowerCase()
    .replace(/[`*_~()[\]{}:;,.!?/\\|<>+"']/g, "")
    .replace(/\s+/g, "-")
    .replace(/-+/g, "-")
    .replace(/^-|-$/g, "");
}

function flattenText(node: ReactNode): string {
  if (typeof node === "string" || typeof node === "number") return String(node);
  if (Array.isArray(node)) return node.map(flattenText).join("");
  if (node && typeof node === "object" && "props" in node) {
    return flattenText((node as { props?: { children?: ReactNode } }).props?.children);
  }
  return "";
}

function prepareMarkdown(raw: string): string {
  // 1. Strip raw <details><summary>Diagram source (Mermaid)</summary> wrappers so our custom code renderer handles them cleanly
  let cleaned = raw.replace(
    /<details>\s*<summary>[^<]*<\/summary>\s*(```mermaid[\s\S]*?```)\s*<\/details>/gi,
    "$1",
  );
  // 2. Convert single-line $$...$$ display math into multi-line fenced math blocks for remark-math v6.
  // A function replacement: in a replacement string "$$" means one literal "$", which turned every displayed
  // formula into inline math.
  cleaned = cleaned.replace(
    /(^|\n)\$\$([^\n$]+?)\$\$(?=\s*(\n|$))/g,
    (_match, _start, body) => `\n\n$$\n${body}\n$$\n\n`,
  );
  return cleaned;
}

function classifyStatus(status: string): "active" | "updated" | "closed" {
  const lower = status.toLowerCase();
  if (lower.includes("closed") || lower.includes("rejected")) return "closed";
  if (lower.includes("superseded") || lower.includes("updated")) return "updated";
  return "active";
}

function parseDecisions(markdown: string): DecisionItem[] {
  const lines = markdown.split("\n");
  let currentTrack = "General";
  const items: DecisionItem[] = [];
  for (const line of lines) {
    const headingMatch = line.match(/^##\s+\d+\.\s+(.+)$/);
    if (headingMatch) {
      currentTrack = headingMatch[1].trim();
      continue;
    }
    if (!line.startsWith("|")) continue;
    const cells = line
      .split("|")
      .slice(1, -1)
      .map((c) => c.trim());
    if (cells.length < 4) continue;
    const idClean = cells[0].replace(/\*/g, "").trim();
    const idMatch = idClean.match(/^D(\d+)$/);
    if (!idMatch) continue;
    const num = Number(idMatch[1]);
    if (cells.length >= 5) {
      const statusRaw = cells[2].replace(/`/g, "").trim();
      items.push({
        id: idClean,
        num,
        date: cells[1],
        status: statusRaw,
        statusKind: classifyStatus(statusRaw),
        track: currentTrack,
        decision: cells[3],
        why: cells[4],
      });
    } else {
      items.push({
        id: idClean,
        num,
        date: cells[1],
        status: "Active",
        statusKind: "active",
        track: currentTrack,
        decision: cells[2],
        why: cells[3],
      });
    }
  }
  return items.sort((a, b) => a.num - b.num);
}

export default function Documentation() {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [selected, setSelected] = useState("README.md");
  const [error, setError] = useState("");
  const [decisionFilterStatus, setDecisionFilterStatus] = useState<string>("all");
  // Click-to-zoom for figures and diagrams. The overlay opens at "fit width" (the figure as wide as the window,
  // scroll for the rest) when that is larger than the whole figure on screen, as for tall diagrams; otherwise at
  // "fit page". + and − zoom by 1.5x, clicking the image switches between the two fits, Escape closes.
  const [zoomed, setZoomed] = useState<{ src: string; alt: string } | null>(null);
  const [zoom, setZoom] = useState(1);
  const fitPage = useRef(0); // image width in px when the whole figure fits the stage
  const widthZoom = useRef(1); // zoom at which the figure is as wide as the stage
  const stageRef = useRef<HTMLDivElement | null>(null);
  const clampZoom = (value: number) => Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, value));
  useEffect(() => {
    if (!zoomed) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") setZoomed(null);
      if (event.key === "+" || event.key === "=") setZoom((value) => clampZoom(value * 1.5));
      if (event.key === "-") setZoom((value) => clampZoom(value / 1.5));
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [zoomed]);
  const [decisionFilterTrack, setDecisionFilterTrack] = useState<string>("all");
  const [decisionSearch, setDecisionSearch] = useState<string>("");
  const [advisorOpen, setAdvisorOpen] = useState(false);

  useEffect(() => {
    const requested = new URL(window.location.href).searchParams.get("document");
    setSelected(requested ?? "docs/report/00_index.md");
    fetch("/research/documentation.json")
      .then(async (response) => {
        if (!response.ok) throw Error("Documentation could not be loaded");
        setDocuments(await response.json());
      })
      .catch((e) => setError(e.message));
  }, []);

  const document =
    documents.find((item) => item.path === selected) ??
    documents.find((item) => item.path === "docs/report/00_index.md") ??
    documents.find((item) => item.path === "README.md") ??
    documents[0];

  function selectDocument(path: string) {
    setSelected(path);
    const url = new URL(window.location.href);
    url.searchParams.set("document", path);
    window.history.replaceState(null, "", url);
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  function relativePath(value: string) {
    return new URL(
      value,
      `https://research.local/${document?.path ?? "README.md"}`,
    ).pathname.slice(1);
  }

  const orderedDocuments = useMemo(() => {
    const byPath = new Map(documents.map((d) => [d.path, d]));
    const result: Array<Document & { meta: DocMeta }> = [];
    for (const meta of DOC_CATALOG) {
      const found = byPath.get(meta.path);
      if (found) {
        result.push({ ...found, meta });
        byPath.delete(meta.path);
      }
    }
    for (const leftover of byPath.values()) {
      result.push({
        ...leftover,
        meta: {
          path: leftover.path,
          shortTitle: leftover.title,
          subtitle: leftover.path,
          category: "4. Architecture & Operations",
        },
      });
    }
    return result;
  }, [documents]);

  const groupedSidebar = useMemo(() => {
    const groups = new Map<string, Array<Document & { meta: DocMeta }>>();
    for (const doc of orderedDocuments) {
      const list = groups.get(doc.meta.category) ?? [];
      list.push(doc);
      groups.set(doc.meta.category, list);
    }
    return Array.from(groups.entries());
  }, [orderedDocuments]);

  const headings = useMemo(() => {
    if (!document) return [];
    const list: Array<{ level: number; text: string; id: string }> = [];
    let inFence = false;
    for (const line of document.content.split("\n")) {
      if (line.trim().startsWith("```")) {
        inFence = !inFence;
        continue;
      }
      if (inFence) continue;
      const match = line.match(/^(#{2,3})\s+(.+)$/);
      if (match) {
        const rawText = match[2]
          .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
          .replace(/\\alpha/g, "α")
          .replace(/\\beta/g, "β")
          .replace(/[`*_~$]/g, "")
          .trim();
        list.push({
          level: match[1].length,
          text: rawText,
          id: slugify(rawText),
        });
      }
    }
    return list;
  }, [document]);

  const decisions = useMemo(() => {
    if (document?.path !== "docs/DECISIONS.md") return [];
    return parseDecisions(document.content);
  }, [document]);

  const decisionTracks = useMemo(() => {
    return Array.from(new Set(decisions.map((d) => d.track)));
  }, [decisions]);

  const filteredDecisions = useMemo(() => {
    const q = decisionSearch.trim().toLowerCase();
    return decisions.filter((d) => {
      if (decisionFilterStatus !== "all" && d.statusKind !== decisionFilterStatus) {
        return false;
      }
      if (decisionFilterTrack !== "all" && d.track !== decisionFilterTrack) {
        return false;
      }
      if (q) {
        const hay = `${d.id} ${d.date} ${d.status} ${d.track} ${d.decision} ${d.why}`.toLowerCase();
        return hay.includes(q);
      }
      return true;
    });
  }, [decisions, decisionFilterStatus, decisionFilterTrack, decisionSearch]);

  const preparedContent = useMemo(
    () => (document ? prepareMarkdown(document.content) : ""),
    [document],
  );

  const currentMeta = orderedDocuments.find((d) => d.path === document?.path)?.meta;

  return (
    <div className="documentation">
      <div className="documentation-header">
        <div>
          <span className="documentation-eyebrow">
            {currentMeta?.category ?? "Research Documentation"}
          </span>
          <h1>Documentation</h1>
          <p className="documentation-subtitle">
            {currentMeta?.subtitle ??
              "Reproducible methods, benchmark logs, architecture, and operational guides for RFL Lab."}
          </p>
        </div>
        <button
          type="button"
          className="advisor-toggle"
          aria-expanded={advisorOpen}
          onClick={() => setAdvisorOpen((open) => !open)}
        >
          {advisorOpen ? "Hide advisor" : "Ask the docs (AI)"}
        </button>
        <label className="documentation-selector">
          Research document
          <select
            value={document?.path ?? selected}
            onChange={(e) => selectDocument(e.target.value)}
          >
            {orderedDocuments.map((item) => (
              <option key={item.path} value={item.path}>
                {item.title}
              </option>
            ))}
          </select>
        </label>
      </div>

      {advisorOpen && documents.length > 0 && (
        <DocAdvisor
          documents={documents}
          currentPath={document?.path ?? selected}
          onClose={() => setAdvisorOpen(false)}
          onOpen={(path, heading) => {
            selectDocument(path);
            window.setTimeout(() => {
              window.document.getElementById(slugify(heading))?.scrollIntoView({ behavior: "smooth", block: "start" });
            }, 400);
          }}
        />
      )}

      {error && <p role="alert">{error}</p>}
      {!document && !error && <p>Loading research Markdown…</p>}

      {document && (
        <div className="documentation-layout">
          <aside className="doc-sidebar" aria-label="Documentation directory">
            {groupedSidebar.map(([category, items]) => (
              <div key={category} className="doc-sidebar-group">
                <h3>{category}</h3>
                <ul>
                  {items.map((item) => {
                    const active = item.path === document.path;
                    return (
                      <li key={item.path}>
                        <button
                          type="button"
                          className={active ? "doc-nav-item active" : "doc-nav-item"}
                          onClick={() => selectDocument(item.path)}
                        >
                          <span className="doc-nav-title">{item.meta.shortTitle}</span>
                          <small className="doc-nav-path">{item.path}</small>
                        </button>
                      </li>
                    );
                  })}
                </ul>
              </div>
            ))}
          </aside>

          <article className="markdown-document">
            <div className="document-meta-bar">
              <span className="document-source">{document.path}</span>
              <span className="document-stats">
                {headings.length} sections ·{" "}
                {Math.max(1, Math.round(document.content.split(/\s+/).length / 220))} min read
              </span>
            </div>

            {document.path === "docs/DECISIONS.md" && decisions.length > 0 && (
              <section
                className="decision-explorer"
                aria-label="Interactive decision explorer"
              >
                <div className="decision-explorer-header">
                  <div>
                    <h2>Interactive Decision Registry (D1–D33)</h2>
                    <p>
                      Filter by lifecycle status, engineering track, or keyword to inspect active invariants versus closed explorations.
                    </p>
                  </div>
                  <div className="decision-counts">
                    <span className="doc-badge active">
                      {decisions.filter((d) => d.statusKind === "active").length} Active
                    </span>
                    <span className="doc-badge updated">
                      {decisions.filter((d) => d.statusKind === "updated").length} Updated / Superseded
                    </span>
                    <span className="doc-badge closed">
                      {decisions.filter((d) => d.statusKind === "closed").length} Closed
                    </span>
                  </div>
                </div>

                <div className="decision-controls">
                  <div className="decision-pills" role="group" aria-label="Filter by status">
                    {[
                      { id: "all", label: `All (${decisions.length})` },
                      {
                        id: "active",
                        label: `Active (${decisions.filter((d) => d.statusKind === "active").length})`,
                      },
                      {
                        id: "updated",
                        label: `Updated / Superseded (${decisions.filter((d) => d.statusKind === "updated").length})`,
                      },
                      {
                        id: "closed",
                        label: `Closed (${decisions.filter((d) => d.statusKind === "closed").length})`,
                      },
                    ].map((pill) => (
                      <button
                        key={pill.id}
                        type="button"
                        className={
                          decisionFilterStatus === pill.id
                            ? "filter-pill active"
                            : "filter-pill"
                        }
                        onClick={() => setDecisionFilterStatus(pill.id)}
                      >
                        {pill.label}
                      </button>
                    ))}
                  </div>

                  <div className="decision-search-row">
                    <input
                      type="search"
                      aria-label="Search decisions"
                      placeholder="Filter D1–D33 by ID, keyword (e.g. ALIE, MAD, backdoor, BatchNorm)…"
                      value={decisionSearch}
                      onChange={(e) => setDecisionSearch(e.target.value)}
                    />
                    { (decisionFilterStatus !== "all" ||
                      decisionFilterTrack !== "all" ||
                      decisionSearch) && (
                      <button
                        type="button"
                        onClick={() => {
                          setDecisionFilterStatus("all");
                          setDecisionFilterTrack("all");
                          setDecisionSearch("");
                        }}
                      >
                        Reset filters
                      </button>
                    )}
                  </div>

                  <div className="decision-pills track-pills" role="group" aria-label="Filter by track">
                    <button
                      type="button"
                      className={
                        decisionFilterTrack === "all"
                          ? "filter-pill active"
                          : "filter-pill"
                      }
                      onClick={() => setDecisionFilterTrack("all")}
                    >
                      All tracks
                    </button>
                    {decisionTracks.map((track) => (
                      <button
                        key={track}
                        type="button"
                        className={
                          decisionFilterTrack === track
                            ? "filter-pill active"
                            : "filter-pill"
                        }
                        onClick={() => setDecisionFilterTrack(track)}
                      >
                        {track}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="decision-cards">
                  {filteredDecisions.map((item) => (
                    <div
                      key={item.id}
                      className={`decision-card status-${item.statusKind}`}
                    >
                      <div className="decision-card-meta">
                        <span className="decision-id">{item.id}</span>
                        <span className={`doc-badge ${item.statusKind}`}>
                          {item.status}
                        </span>
                        <span className="decision-track-tag">{item.track}</span>
                        <time className="decision-date">{item.date}</time>
                      </div>
                      <div className="decision-card-body">
                        <div className="decision-main">
                          <strong>Decision:</strong>{" "}
                          <Markdown
                            remarkPlugins={[remarkGfm, remarkMath]}
                            rehypePlugins={[rehypeKatex]}
                            components={{
                              p: ({ children }) => <span>{children}</span>,
                            }}
                          >
                            {item.decision}
                          </Markdown>
                        </div>
                        <div className="decision-why">
                          <strong>Rationale:</strong>{" "}
                          <Markdown
                            remarkPlugins={[remarkGfm, remarkMath]}
                            rehypePlugins={[rehypeKatex]}
                            components={{
                              p: ({ children }) => <span>{children}</span>,
                            }}
                          >
                            {item.why}
                          </Markdown>
                        </div>
                      </div>
                    </div>
                  ))}
                  {filteredDecisions.length === 0 && (
                    <p className="decision-empty">
                      No decisions match your current filter criteria.
                    </p>
                  )}
                </div>
              </section>
            )}

            <Markdown
              remarkPlugins={[remarkGfm, remarkMath]}
              rehypePlugins={[rehypeKatex]}
              components={{
                h2: ({ children }) => {
                  const text = flattenText(children);
                  return <h2 id={slugify(text)}>{children}</h2>;
                },
                h3: ({ children }) => {
                  const text = flattenText(children);
                  return <h3 id={slugify(text)}>{children}</h3>;
                },
                a: ({ href, children }) => {
                  if (!href) return <a>{children}</a>;
                  const resolved = relativePath(href).split("#")[0];
                  const match = documents.find((item) => item.path === resolved);
                  if (match) {
                    return (
                      <a
                        href={`?tab=Documentation&document=${encodeURIComponent(match.path)}`}
                        onClick={(e) => {
                          e.preventDefault();
                          selectDocument(match.path);
                        }}
                      >
                        {children}
                      </a>
                    );
                  }
                  if (resolved.endsWith(".html")) {
                    return (
                      <a
                        href={`/research/documents/${resolved}`}
                        target="_blank"
                        rel="noreferrer"
                      >
                        {children}
                      </a>
                    );
                  }
                  return <a href={href}>{children}</a>;
                },
                img: ({ src, alt }) => {
                  const url =
                    typeof src === "string" && !/^https?:|^data:/.test(src)
                      ? `/research/documents/${relativePath(src)}`
                      : typeof src === "string"
                        ? src
                        : "";
                  const label = alt ?? "Research figure";
                  const open = () => {
                    fitPage.current = 0;
                    setZoom(1);
                    setZoomed({ src: url, alt: label });
                  };
                  return (
                    <img
                      className="doc-img zoomable"
                      src={url}
                      alt={label}
                      title="Click to enlarge"
                      loading="lazy"
                      role="button"
                      tabIndex={0}
                      onClick={open}
                      onKeyDown={(event) => {
                        if (event.key === "Enter" || event.key === " ") {
                          event.preventDefault();
                          open();
                        }
                      }}
                    />
                  );
                },
                pre: ({ children }) => {
                  const childProps =
                    children && typeof children === "object" && "props" in children
                      ? (children as { props?: { className?: string } }).props
                      : undefined;
                  if (childProps?.className?.includes("language-mermaid")) {
                    return (
                      <details
                        className="mermaid-details"
                        open={document.path === "docs/ARCHITECTURE.md"}
                      >
                        <summary>Mermaid flow specification</summary>
                        <pre className="mermaid-code">{children}</pre>
                      </details>
                    );
                  }
                  return <pre>{children}</pre>;
                },
                code: ({ className, children }) => {
                  const text = String(children).replace(/\n$/, "");
                  if (!className) {
                    const lower = text.toLowerCase();
                    if (
                      lower === "active" ||
                      lower.startsWith("updated →") ||
                      lower.startsWith("superseded →") ||
                      lower.startsWith("closed (")
                    ) {
                      const kind = classifyStatus(text);
                      return <span className={`doc-badge ${kind}`}>{text}</span>;
                    }
                  }
                  return <code className={className}>{children}</code>;
                },
                table: ({ children }) => (
                  <div className="table-scroll">
                    <table>{children}</table>
                  </div>
                ),
              }}
            >
              {preparedContent}
            </Markdown>
            {zoomed && (
              <div
                className="doc-lightbox"
                role="dialog"
                aria-modal="true"
                aria-label={zoomed.alt}
                onClick={() => setZoomed(null)}
              >
                <div className="doc-lightbox-bar" onClick={(event) => event.stopPropagation()}>
                  <span className="doc-lightbox-title">{zoomed.alt}</span>
                  <button
                    type="button"
                    aria-label="Zoom out"
                    disabled={zoom <= ZOOM_MIN}
                    onClick={() => setZoom((value) => clampZoom(value / 1.5))}
                  >
                    −
                  </button>
                  <span className="doc-lightbox-scale">{Math.round(zoom * 100)}%</span>
                  <button
                    type="button"
                    aria-label="Zoom in"
                    disabled={zoom >= ZOOM_MAX}
                    onClick={() => setZoom((value) => clampZoom(value * 1.5))}
                  >
                    +
                  </button>
                  <button type="button" onClick={() => setZoom(1)}>
                    Fit page
                  </button>
                  <button type="button" onClick={() => setZoom(clampZoom(widthZoom.current))}>
                    Fit width
                  </button>
                  <a href={zoomed.src} target="_blank" rel="noreferrer">
                    Open in new tab
                  </a>
                  <button type="button" onClick={() => setZoomed(null)} aria-label="Close enlarged figure">
                    Close
                  </button>
                </div>
                <div
                  ref={stageRef}
                  className={zoom > 1 ? "doc-lightbox-stage zoomed" : "doc-lightbox-stage"}
                >
                  <img
                    src={zoomed.src}
                    alt={zoomed.alt}
                    style={zoom > 1 && fitPage.current ? { width: fitPage.current * zoom } : undefined}
                    onLoad={(event) => {
                      const stage = stageRef.current;
                      const image = event.currentTarget;
                      if (!stage || fitPage.current) return;
                      fitPage.current = image.clientWidth;
                      const usable = stage.clientWidth - 32;
                      widthZoom.current = image.clientWidth ? usable / image.clientWidth : 1;
                      if (widthZoom.current > 1.3) setZoom(clampZoom(widthZoom.current));
                    }}
                    onClick={(event) => {
                      event.stopPropagation();
                      setZoom((value) => (value > 1 ? 1 : clampZoom(Math.max(widthZoom.current, 2))));
                    }}
                  />
                </div>
              </div>
            )}
          </article>

          {headings.length > 1 && (
            <aside className="doc-toc" aria-label="On this page">
              <h3>On this page</h3>
              <ul>
                {headings.map((h, idx) => (
                  <li
                    key={`${h.id}-${idx}`}
                    className={h.level === 3 ? "toc-sub" : "toc-main"}
                  >
                    <a
                      href={`#${h.id}`}
                      onClick={(e) => {
                        e.preventDefault();
                        const target = window.document.getElementById(h.id);
                        if (target) {
                          target.scrollIntoView({ behavior: "smooth", block: "start" });
                        }
                      }}
                    >
                      {h.text}
                    </a>
                  </li>
                ))}
              </ul>
            </aside>
          )}
        </div>
      )}
    </div>
  );
}
