"use client";
// AI advisor for the Documentation tab. Runs entirely in the browser against the reader's own Ollama server
// (default http://localhost:11434): the documents are split by heading, embedded with an embedding model
// (bge-m3), the closest passages to a question are retrieved, and a chat model answers from them with
// numbered citations. Nothing leaves the reader's machine; the deployed site only hosts the page.
import { useEffect, useMemo, useRef, useState } from "react";
import Markdown from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";

type Document = { path: string; title: string; content: string };
type Chunk = { path: string; title: string; heading: string; text: string };
type Turn = { role: "user" | "assistant"; content: string; sources?: Chunk[] };
type Settings = { url: string; chat: string; embed: string };

const DEFAULTS: Settings = { url: "http://localhost:11434", chat: "qwen3.8:27b", embed: "bge-m3" };
const CHUNK_CHARS = 1600;
const TOP_K = 6;

const SYSTEM = `You are the research advisor for RFL Lab, a robust federated learning project (FedHEP, FedHEP+, FedHEP-Guard).
Answer the reader's question using only the numbered passages from the project's documentation below.
Explain in plain language first, then give the precise detail (numbers, settings, formulas) the passages support.
Cite passages inline as [1], [2]. If the passages do not contain the answer, say so plainly and say which document is most likely to cover it; never invent results, numbers or citations.
Use Markdown; write formulas in LaTeX between $ signs.`;

function load(): Settings {
  try {
    return { ...DEFAULTS, ...JSON.parse(localStorage.getItem("rfl-advisor") ?? "{}") };
  } catch {
    return DEFAULTS;
  }
}

function save(settings: Settings) {
  try {
    localStorage.setItem("rfl-advisor", JSON.stringify(settings));
  } catch {
    /* storage unavailable: settings last for this visit only */
  }
}

/** Split each document at its headings; long sections are cut at paragraph breaks. */
export function chunkDocuments(documents: Document[]): Chunk[] {
  const chunks: Chunk[] = [];
  for (const doc of documents) {
    let heading = doc.title;
    let buffer: string[] = [];
    const flush = () => {
      const text = buffer.join("\n").trim();
      buffer = [];
      if (text.length < 40) return;
      let piece = "";
      for (const paragraph of text.split(/\n\s*\n/)) {
        if (piece && piece.length + paragraph.length > CHUNK_CHARS) {
          chunks.push({ path: doc.path, title: doc.title, heading, text: piece.trim() });
          piece = "";
        }
        piece += paragraph + "\n\n";
      }
      if (piece.trim()) chunks.push({ path: doc.path, title: doc.title, heading, text: piece.trim() });
    };
    for (const line of doc.content.split("\n")) {
      const match = line.match(/^#{1,4}\s+(.+)$/);
      if (match) {
        flush();
        heading = match[1].replace(/[`*_]/g, "").trim();
      }
      buffer.push(line);
    }
    flush();
  }
  return chunks;
}

function cosine(a: Float32Array, b: Float32Array) {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na * nb) || 1);
}

async function hash(text: string) {
  const bytes = new TextEncoder().encode(text);
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(digest).slice(0, 8), (b) => b.toString(16).padStart(2, "0")).join("");
}

async function embed(settings: Settings, input: string[], signal?: AbortSignal): Promise<Float32Array[]> {
  const response = await fetch(`${settings.url}/api/embed`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model: settings.embed, input }),
    signal,
  });
  if (!response.ok) throw Error(`Embedding failed (${response.status}): ${await response.text()}`);
  const body = (await response.json()) as { embeddings: number[][] };
  return body.embeddings.map((v) => Float32Array.from(v));
}

export default function DocAdvisor({
  documents,
  currentPath,
  onOpen,
  onClose,
}: {
  documents: Document[];
  currentPath: string;
  onOpen: (path: string, heading: string) => void;
  onClose: () => void;
}) {
  const [settings, setSettings] = useState<Settings>(DEFAULTS);
  const [models, setModels] = useState<string[]>([]);
  const [status, setStatus] = useState<"checking" | "online" | "offline">("checking");
  const [problem, setProblem] = useState("");
  const [showSettings, setShowSettings] = useState(false);
  const [progress, setProgress] = useState("");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState(false);
  const index = useRef<{ key: string; vectors: Float32Array[] } | null>(null);
  const abort = useRef<AbortController | null>(null);
  const bottom = useRef<HTMLDivElement | null>(null);
  const chunks = useMemo(() => chunkDocuments(documents), [documents]);

  useEffect(() => {
    const saved = load();
    setSettings(saved);
    connect(saved);
    // Connect once on opening; Retry and the address field reconnect later.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function connect(current: Settings) {
    setStatus("checking");
    setProblem("");
    try {
      const response = await fetch(`${current.url}/api/tags`);
      if (!response.ok) throw Error(`Ollama answered ${response.status}`);
      const body = (await response.json()) as { models: Array<{ name: string }> };
      const names = body.models.map((m) => m.name);
      setModels(names);
      // Installed names carry a tag (bge-m3:latest); keep the reader's choice if installed, else the closest one.
      const pick = (want: string, fallback: (n: string) => boolean) =>
        names.find((n) => n === want || n === `${want}:latest`) ??
        names.find((n) => n.startsWith(want.split(":")[0])) ??
        names.find(fallback) ??
        want;
      const next = {
        ...current,
        chat: pick(current.chat, (n) => !/embed|bge|nomic|minilm/i.test(n)),
        embed: pick(current.embed, (n) => /embed|bge|nomic|minilm/i.test(n)),
      };
      setSettings(next);
      save(next);
      setStatus("online");
    } catch {
      setStatus("offline");
      const remote = window.location.hostname !== "localhost" && window.location.hostname !== "127.0.0.1";
      setProblem(
        remote
          ? `No answer from Ollama at ${current.url}. This page is served from ${window.location.origin}, so Ollama must allow it: quit Ollama, run \`launchctl setenv OLLAMA_ORIGINS "${window.location.origin}"\` (macOS) or start it with \`OLLAMA_ORIGINS=${window.location.origin} ollama serve\`, then open Ollama again. Chrome may also ask to allow access to devices on your local network. Safari blocks this; use Chrome, Firefox, or the visualizer on localhost.`
          : `No answer from Ollama at ${current.url}. Start the Ollama app (or run \`ollama serve\`) and press Retry.`,
      );
    }
  }

  useEffect(() => {
    // Braces matter: newer browsers return a promise from scrollIntoView, which React would call as a cleanup.
    bottom.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [turns]);

  async function ensureIndex(signal: AbortSignal) {
    const key = await hash(settings.embed + chunks.map((c) => c.path + c.text).join("\u0000"));
    if (index.current?.key === key) return index.current.vectors;
    try {
      const cached = JSON.parse(localStorage.getItem(`rfl-advisor-index-${key}`) ?? "null") as string[] | null;
      if (cached && cached.length === chunks.length) {
        const vectors = cached.map((b64) => new Float32Array(Uint8Array.from(atob(b64), (c) => c.charCodeAt(0)).buffer));
        index.current = { key, vectors };
        return vectors;
      }
    } catch {
      /* no usable cache: embed again */
    }
    const vectors: Float32Array[] = [];
    for (let i = 0; i < chunks.length; i += 24) {
      setProgress(`Reading the documentation: ${Math.min(i + 24, chunks.length)} of ${chunks.length} passages`);
      const batch = chunks.slice(i, i + 24).map((c) => `${c.title} › ${c.heading}\n${c.text}`);
      vectors.push(...(await embed(settings, batch, signal)));
    }
    setProgress("");
    index.current = { key, vectors };
    try {
      for (let k = 0; k < localStorage.length; k++) {
        const name = localStorage.key(k);
        if (name?.startsWith("rfl-advisor-index-")) localStorage.removeItem(name);
      }
      const encoded = vectors.map((v) => btoa(String.fromCharCode(...new Uint8Array(v.buffer))));
      localStorage.setItem(`rfl-advisor-index-${key}`, JSON.stringify(encoded));
    } catch {
      /* too large or storage unavailable: the index lasts for this visit */
    }
    return vectors;
  }

  async function ask(text: string) {
    const q = text.trim();
    if (!q || busy) return;
    setQuestion("");
    setBusy(true);
    const controller = new AbortController();
    abort.current = controller;
    const history = turns.slice(-4);
    setTurns((t) => [...t, { role: "user", content: q }, { role: "assistant", content: "" }]);
    try {
      const vectors = await ensureIndex(controller.signal);
      const [query] = await embed(settings, [q], controller.signal);
      const ranked = chunks
        .map((c, i) => ({ c, score: cosine(query, vectors[i]) + (c.path === currentPath ? 0.03 : 0) }))
        .sort((a, b) => b.score - a.score)
        .slice(0, TOP_K)
        .map((r) => r.c);
      const context = ranked
        .map((c, i) => `[${i + 1}] ${c.path} › ${c.heading}\n${c.text}`)
        .join("\n\n---\n\n");
      setTurns((t) => [...t.slice(0, -1), { role: "assistant", content: "", sources: ranked }]);
      const messages = [
        { role: "system", content: `${SYSTEM}\n\nThe reader is currently viewing ${currentPath}.\n\nPassages:\n\n${context}` },
        ...history.map(({ role, content }) => ({ role, content })),
        { role: "user", content: q },
      ];
      let response = await fetch(`${settings.url}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: settings.chat, messages, stream: true, think: false }),
        signal: controller.signal,
      });
      if (response.status === 400) {
        // Models without a thinking switch reject `think`; ask again without it.
        response = await fetch(`${settings.url}/api/chat`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ model: settings.chat, messages, stream: true }),
          signal: controller.signal,
        });
      }
      if (!response.ok || !response.body) throw Error(`Chat failed (${response.status}): ${await response.text()}`);
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let pending = "";
      let answer = "";
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        pending += decoder.decode(value, { stream: true });
        const lines = pending.split("\n");
        pending = lines.pop() ?? "";
        for (const line of lines) {
          if (!line.trim()) continue;
          const part = JSON.parse(line) as { message?: { content?: string }; error?: string };
          if (part.error) throw Error(part.error);
          answer += part.message?.content ?? "";
          const shown = answer.replace(/<think>[\s\S]*?(<\/think>|$)/g, "").trimStart();
          setTurns((t) => [...t.slice(0, -1), { ...t[t.length - 1], content: shown }]);
        }
      }
    } catch (e) {
      const message = (e as Error).name === "AbortError" ? "_Stopped._" : `**Could not answer:** ${(e as Error).message}`;
      setTurns((t) => [...t.slice(0, -1), { ...t[t.length - 1], content: (t[t.length - 1].content || "") + `\n\n${message}` }]);
    } finally {
      setBusy(false);
      setProgress("");
      abort.current = null;
    }
  }

  function askAboutSelection() {
    const selected = window.getSelection()?.toString().trim();
    if (selected) setQuestion(`Explain this passage in plain words: "${selected.slice(0, 600)}"`);
  }

  return (
    <aside className="advisor" aria-label="AI advisor">
      <header className="advisor-head">
        <div>
          <h3>Ask the docs</h3>
          <span className={`advisor-status ${status}`}>
            {status === "online" ? `Ollama · ${settings.chat}` : status === "checking" ? "Connecting to Ollama…" : "Ollama not reachable"}
          </span>
        </div>
        <div className="advisor-actions">
          <button type="button" onClick={() => setShowSettings((s) => !s)} aria-expanded={showSettings}>
            Settings
          </button>
          <button type="button" onClick={onClose} aria-label="Close advisor">
            Close
          </button>
        </div>
      </header>

      {showSettings && (
        <div className="advisor-settings">
          <label>
            Ollama address
            <input
              value={settings.url}
              onChange={(e) => setSettings({ ...settings, url: e.target.value.replace(/\/$/, "") })}
              onBlur={() => {
                save(settings);
                connect(settings);
              }}
            />
          </label>
          <label>
            Answer model
            <select value={settings.chat} onChange={(e) => { const s = { ...settings, chat: e.target.value }; setSettings(s); save(s); }}>
              {[...new Set([settings.chat, ...models])].map((m) => <option key={m}>{m}</option>)}
            </select>
          </label>
          <label>
            Embedding model
            <select value={settings.embed} onChange={(e) => { const s = { ...settings, embed: e.target.value }; setSettings(s); save(s); index.current = null; }}>
              {[...new Set([settings.embed, ...models])].map((m) => <option key={m}>{m}</option>)}
            </select>
          </label>
          <p className="advisor-note">
            Runs on your own machine: the documentation and your questions go only to the Ollama server above.
            {chunks.length} passages from {documents.length} documents.
          </p>
        </div>
      )}

      {status === "offline" && (
        <div className="advisor-problem" role="alert">
          <Markdown>{problem}</Markdown>
          <button type="button" onClick={() => connect(settings)}>Retry</button>
        </div>
      )}

      <div className="advisor-log">
        {turns.length === 0 && (
          <div className="advisor-empty">
            <p>Ask anything about the project; answers cite the passages they come from.</p>
            {[
              "Why does the backdoor not lower clean accuracy?",
              "What does the gap gate do, and why gate 12?",
              "Explain L0 (ARC, NNM, trimmed mean) simply.",
            ].map((s) => (
              <button key={s} type="button" className="advisor-suggestion" onClick={() => ask(s)} disabled={status !== "online"}>
                {s}
              </button>
            ))}
          </div>
        )}
        {turns.map((turn, i) => (
          <div key={i} className={`advisor-turn ${turn.role}`}>
            {turn.role === "user" ? (
              <p>{turn.content}</p>
            ) : (
              <>
                <Markdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]}>
                  {turn.content || (busy && i === turns.length - 1 ? progress || "Thinking…" : "")}
                </Markdown>
                {turn.sources && turn.sources.length > 0 && (
                  <ol className="advisor-sources">
                    {turn.sources.map((s, k) => (
                      <li key={k}>
                        <button type="button" onClick={() => onOpen(s.path, s.heading)}>
                          [{k + 1}] {s.title} › {s.heading}
                        </button>
                      </li>
                    ))}
                  </ol>
                )}
              </>
            )}
          </div>
        ))}
        <div ref={bottom} />
      </div>

      <form
        className="advisor-input"
        onSubmit={(e) => {
          e.preventDefault();
          ask(question);
        }}
      >
        <textarea
          rows={3}
          value={question}
          placeholder={status === "online" ? "Ask about a method, a result, a decision…" : "Connect Ollama to ask"}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              ask(question);
            }
          }}
        />
        <div className="advisor-input-actions">
          <button type="button" onClick={askAboutSelection} title="Select text in the document first">
            Explain selection
          </button>
          {busy ? (
            <button type="button" onClick={() => abort.current?.abort()}>Stop</button>
          ) : (
            <button type="submit" disabled={status !== "online" || !question.trim()}>Ask</button>
          )}
        </div>
      </form>
    </aside>
  );
}
