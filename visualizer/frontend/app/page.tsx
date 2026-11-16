"use client";
import { useEffect, useMemo, useState, useRef } from "react";
import dynamic from "next/dynamic";
import Documentation from "./Documentation";
import {
  edgeTypes,
  ExecutionAnimation,
  eventFocus,
  eventPhase,
  isPlaybackStep,
} from "./ExecutionAnimation";
import {
  ReactFlow,
  useReactFlow,
  useNodesInitialized,
  Controls,
  Background,
  MiniMap,
  MarkerType,
  Position,
  Node,
  Edge,
} from "@xyflow/react";
import { MotionConfig, motion } from "framer-motion";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";
import Markdown from "react-markdown";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import {
  ClientDistribution,
  RoundWalkthrough,
  ExperimentHistory,
} from "./ResearchViews";
const LogTerminal = dynamic(() => import("./Terminal"), { ssr: false });
type Event = {
  type: string;
  round?: number;
  client?: number | string;
  timestamp: number;
  elapsed_s?: number;
  [key: string]: any;
};
type Field = { key: string; default: any; options: any[] | null; type: string };
async function api(path: string, body?: unknown) {
  const key =
    localStorage.getItem("rfl-cpu-key") ??
    sessionStorage.getItem("rfl-cpu-key");
  const endpoint =
    localStorage.getItem("rfl-cpu-url") ??
    sessionStorage.getItem("rfl-cpu-url");
  const resource = path.split("?")[0];
  const staticPath =
    resource === "/history"
      ? "/research/history.json"
      : resource === "/documentation"
        ? "/research/documentation.json"
        : resource === "/live/captures"
          ? "/research/captures.json"
          : resource.startsWith("/live/captures/")
            ? `/research/captures/${resource.split("/").at(-1)}.json`
            : null;
  const response = await fetch(staticPath ?? "/api" + path, {
    method: body === undefined ? "GET" : "POST",
    headers: {
      "Content-Type": "application/json",
      ...(!staticPath && key ? { "X-RFL-Key": key } : {}),
      ...(!staticPath && endpoint ? { "X-RFL-Endpoint": endpoint } : {}),
    },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const value = await response.json();
  if (!response.ok)
    throw Error(
      typeof value.detail === "string"
        ? value.detail
        : JSON.stringify(value.detail),
    );
  return value;
}
const pretty = (value: unknown) => JSON.stringify(value, null, 2);
const deviceLabels: Record<string, string> = {
  cpu: "CPU",
  mps: "MPS · Mac GPU",
  cuda: "CUDA / ROCm GPU",
  xpu: "XPU · Intel GPU",
  auto: "Auto · available accelerator",
};
const titles: Record<string, string> = {
  fedavg: "FedAvg",
  fedprox: "FedProx",
  scaffold: "SCAFFOLD",
  multikrum: "Multi-Krum",
  ditto: "Ditto",
  fedrep: "FedRep",
  fedhep: "FedHEP",
  fedhep_defended: "Defended FedHEP",
  fedhep_plus: "FedHEP+",
  fedhep_robust: "Robust FedHEP",
};
function AutoFit({ focus }: { focus: string }) {
  const { fitView, getNodes } = useReactFlow();
  const nodesReady = useNodesInitialized();
  const hasFitted = useRef(false);
  useEffect(() => {
    const onResize = () =>
      requestAnimationFrame(() => fitView({ padding: 0.2 }));
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, [fitView]);
  useEffect(() => {
    if (!nodesReady) return;
    const frame = requestAnimationFrame(() => {
      const ids = new Set(focus.split("|"));
      const targets = getNodes().filter((node) => ids.has(node.id));
      void fitView({
        ...(targets.length ? { nodes: targets } : {}),
        maxZoom: 1.2,
        padding: 0.3,
        duration:
          hasFitted.current &&
          !window.matchMedia("(prefers-reduced-motion: reduce)").matches
            ? 300
            : 0,
      });
      hasFitted.current = true;
    });
    return () => cancelAnimationFrame(frame);
  }, [focus, nodesReady, fitView, getNodes]);
  return null;
}

export default function Workbench() {
  const [capabilities, setCapabilities] = useState({
    training: true,
    online: true,
    requiresKey: false,
    devices: ["cpu"] as string[],
    mode: "Local Python execution",
  });
  const [method, setMethod] = useState("fedhep_plus"),
    [models, setModels] = useState<any[]>([]),
    [fields, setFields] = useState<Field[]>([]),
    [values, setValues] = useState<Record<string, any>>({});
  const [view, setView] = useState("Live Workbench"),
    [dark, setDark] = useState(true),
    [error, setError] = useState(""),
    [id, setId] = useState(""),
    [status, setStatus] = useState("Ready"),
    [busy, setBusy] = useState(false);
  const [events, setEvents] = useState<Event[]>([]),
    [history, setHistory] = useState<any[]>([]),
    [formulas, setFormulas] = useState<any[]>([]),
    [selected, setSelected] = useState("global"),
    [cursor, setCursor] = useState<number | null>(null),
    [filter, setFilter] = useState(""),
    [clearAt, setClearAt] = useState(0),
    [terminalOpen, setTerminalOpen] = useState(true),
    [source, setSource] = useState("");
  const [captures, setCaptures] = useState<any[]>([]);
  const [captureId, setCaptureId] = useState("fedhep_plus-sign_flip");
  const [captureMode, setCaptureMode] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [pageVisible, setPageVisible] = useState(true);
  const [speed, setSpeed] = useState(1);
  const [followCamera, setFollowCamera] = useState(true);
  const [focus, setFocus] = useState("");
  const [sidebarHidden, setSidebarHidden] = useState(false);
  const [historyTarget, setHistoryTarget] = useState("");
  const [cpuKey, setCpuKey] = useState("");
  const [cpuEndpoint, setCpuEndpoint] = useState("");
  const configurationRequest = useRef(0);
  const [schemaLoading, setSchemaLoading] = useState(false);
  const stream = useRef<{ close: () => void } | null>(null);
  const buffer = useRef<Event[]>([]);
  useEffect(() => {
    const shared = () => {
      const url = new URL(window.location.href);
      const target = url.searchParams.get("experiment") ?? "";
      setHistoryTarget(target);
      const tab = url.searchParams.get("tab");
      setView(
        [
          "Live Workbench",
          "Architecture",
          "Mathematical inspection",
          "Experiment history",
          "Documentation",
        ].includes(tab ?? "")
          ? tab!
          : target
            ? "Experiment history"
            : "Live Workbench",
      );
      const key = new URLSearchParams(url.hash.slice(1)).get("cpu-key");
      const endpoint = new URLSearchParams(url.hash.slice(1)).get("cpu-api");
      if (endpoint && key) {
        sessionStorage.setItem("rfl-cpu-url", endpoint);
        localStorage.setItem("rfl-cpu-url", endpoint);
      }
      if (key) {
        sessionStorage.setItem("rfl-cpu-key", key);
        localStorage.setItem("rfl-cpu-key", key);
        url.hash = "";
        window.history.replaceState(null, "", url);
      }
      setCpuKey(
        localStorage.getItem("rfl-cpu-key") ??
          sessionStorage.getItem("rfl-cpu-key") ??
          "",
      );
      setCpuEndpoint(
        localStorage.getItem("rfl-cpu-url") ??
          sessionStorage.getItem("rfl-cpu-url") ??
          "",
      );
      const storedKey =
        localStorage.getItem("rfl-cpu-key") ??
        sessionStorage.getItem("rfl-cpu-key");
      const storedEndpoint =
        localStorage.getItem("rfl-cpu-url") ??
        sessionStorage.getItem("rfl-cpu-url");
      if (storedKey && storedEndpoint) {
        localStorage.setItem("rfl-cpu-key", storedKey);
        localStorage.setItem("rfl-cpu-url", storedEndpoint);
      }
    };
    shared();
    const initialUrl = new URL(window.location.href);
    const savedCapture = initialUrl.searchParams.get("capture");
    const savedRun = initialUrl.searchParams.get("run");
    if (savedRun)
      api(`/experiments/${savedRun}`)
        .then(async (run) => {
          await installConfiguration(run.config);
          setStatus(run.status);
          connect(run.id, Number(initialUrl.searchParams.get("step") ?? 0));
        })
        .catch((error) => setError(error.message));
    else if (savedCapture)
      loadCapture(
        savedCapture,
        Number(initialUrl.searchParams.get("step") ?? 0),
        initialUrl.searchParams.get("tab") ?? "Live Workbench",
      );
    else
      installConfiguration({ method: "fedhep_plus", overrides: {} }).catch(
        (e) => setError(e.message),
      );
    setSidebarHidden(localStorage.getItem("rfl-sidebar-hidden") === "true");
    window.addEventListener("popstate", shared);
    api("/capabilities")
      .then((c) => {
        setCapabilities(c);
      })
      .catch((e) => setError(e.message));
    Promise.all([api("/models"), api("/formulas")])
      .then(([m, f]) => {
        setModels(m);
        setFormulas(f);
      })
      .catch((e) => setError(e.message));
    api("/history")
      .then(setHistory)
      .catch((e) => setError(e.message));
    api("/live/captures")
      .then(setCaptures)
      .catch((e) => setError(e.message));
    setDark(localStorage.getItem("rfl-theme") !== "light");
    const refresh = setInterval(
      () =>
        api("/history?refresh=true")
          .then(setHistory)
          .catch((e) => setError(e.message)),
      900000,
    );
    const health = setInterval(
      () =>
        api("/capabilities")
          .then(setCapabilities)
          .catch(() => {}),
      15000,
    );
    return () => {
      stream.current?.close();
      clearInterval(refresh);
      clearInterval(health);
      window.removeEventListener("popstate", shared);
      configurationRequest.current += 1;
    };
  }, []);
  useEffect(() => {
    document.documentElement.dataset.theme = dark ? "dark" : "light";
    localStorage.setItem("rfl-theme", dark ? "dark" : "light");
  }, [dark]);
  useEffect(() => {
    const timer = setInterval(() => {
      if (buffer.current.length) {
        const batch = buffer.current.splice(0);
        setEvents((old) => [...old, ...batch]);
      }
    }, 100);
    return () => clearInterval(timer);
  }, []);
  function connect(runId: string, restoredStep = 0) {
    const url = new URL(window.location.href);
    url.searchParams.set("run", runId);
    url.searchParams.delete("capture");
    window.history.replaceState(null, "", url);
    stream.current?.close();
    buffer.current = [];
    setEvents([]);
    setCursor(Math.max(0, restoredStep));
    setClearAt(0);
    setId(runId);
    let closed = false;
    let after = 0;
    let timer: ReturnType<typeof setTimeout>;
    stream.current = {
      close: () => {
        closed = true;
        clearTimeout(timer);
      },
    };
    async function poll() {
      try {
        const batch = await api(
          `/experiments/${runId}/event-batch?after=${after}`,
        );
        if (closed) return;
        after = batch.cursor;
        buffer.current.push(...batch.events);
        setStatus(batch.status);
        if (
          batch.more ||
          ["running", "paused", "cancelling", "queued", "pending"].includes(
            batch.status,
          )
        )
          timer = setTimeout(poll, batch.more ? 0 : 1000);
      } catch (e) {
        if (closed) return;
        setError((e as Error).message);
        timer = setTimeout(poll, 5000);
      }
    }
    void poll();
  }
  async function start() {
    setBusy(true);
    setError("");
    setCaptureMode(false);
    setPlaying(false);
    try {
      const request = { method, overrides: values };
      await api("/experiments/validate", request);
      const run = await api("/experiments", request);
      setStatus(run.status);
      connect(run.id);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  async function installConfiguration(config: any) {
    const request = ++configurationRequest.current;
    setSchemaLoading(true);
    setFields([]);
    try {
      const schema = await api(`/models/${config.method}/schema`);
      if (request !== configurationRequest.current) return false;
      setMethod(config.method);
      setFields(schema);
      setValues(
        Object.fromEntries(
          schema.map((field: Field) => [
            field.key,
            field.key in config.overrides
              ? config.overrides[field.key]
              : field.default,
          ]),
        ),
      );
      return true;
    } finally {
      if (request === configurationRequest.current) setSchemaLoading(false);
    }
  }
  async function loadCapture(
    selectedId = captureId,
    restoredStep?: number,
    restoredView = "Live Workbench",
  ) {
    setBusy(true);
    setError("");
    stream.current?.close();
    buffer.current = [];
    try {
      const capture = await api(`/live/captures/${selectedId}`);
      if (!(await installConfiguration(capture.config))) return;
      setCaptureMode(true);
      setCaptureId(selectedId);
      setPlaying(false);
      setEvents(capture.events);
      setId(capture.id);
      setCursor(
        Math.max(
          0,
          restoredStep === undefined
            ? capture.events.findIndex((e: Event) => e.type === "initialized")
            : Math.min(restoredStep, capture.events.length - 1),
        ),
      );
      setClearAt(0);
      setStatus("Visualization playback");
      navigate(restoredView);
      const url = new URL(window.location.href);
      url.searchParams.set("capture", selectedId);
      url.searchParams.delete("run");
      window.history.replaceState(null, "", url);
      setFocus("");
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  function navigate(next: string) {
    if (next !== "Live Workbench") setPlaying(false);
    setView(next);
    const url = new URL(window.location.href);
    url.searchParams.set("tab", next);
    if (next !== "Experiment history") {
      url.searchParams.delete("experiment");
      url.searchParams.delete("section");
    }
    window.history.pushState(null, "", url);
  }
  useEffect(() => {
    if (!id || cursor === null) return;
    const url = new URL(window.location.href);
    url.searchParams.set("step", String(cursor));
    window.history.replaceState(null, "", url);
  }, [id, cursor]);
  useEffect(() => {
    const visibility = () => {
      setPageVisible(!document.hidden);
      if (document.hidden) setPlaying(false);
    };
    document.addEventListener("visibilitychange", visibility);
    visibility();
    return () => document.removeEventListener("visibilitychange", visibility);
  }, []);
  useEffect(() => {
    if (view === "Live Workbench" && pageVisible) return;
    setPlaying(false);
    if (
      !captureMode &&
      id &&
      ["running", "queued", "pending"].includes(status)
    ) {
      void api(`/experiments/${id}/pause`, {})
        .then((run) => setStatus(run.status))
        .catch((error) =>
          setError(`Could not pause local training. ${error.message}`),
        );
    }
  }, [view, pageVisible, captureMode, id, status]);
  const nextPlayback = events.findIndex(
    (e, i) => i > (cursor ?? -1) && isPlaybackStep(e),
  );
  const playbackDelay =
    events[cursor ?? 0]?.type === "operation_completed" ? 90 : 600;
  useEffect(() => {
    if (!playing || view !== "Live Workbench" || !pageVisible) return;
    const next = nextPlayback;
    if (next < 0) {
      if (!["running", "pending", "queued", "cancelling"].includes(status))
        setPlaying(false);
      return;
    }
    const timer = setTimeout(() => setCursor(next), playbackDelay / speed);
    return () => clearTimeout(timer);
  }, [
    playing,
    cursor,
    nextPlayback,
    playbackDelay,
    speed,
    status,
    view,
    pageVisible,
  ]);
  function seek(index: number) {
    setPlaying(false);
    setCursor(index);
    if (followCamera && events[index]) setFocus(eventFocus(events[index]));
  }
  function resetInspection() {
    const url = new URL(window.location.href);
    url.searchParams.delete("capture");
    url.searchParams.delete("run");
    url.searchParams.delete("step");
    window.history.replaceState(null, "", url);
    stream.current?.close();
    buffer.current = [];
    setEvents([]);
    setId("");
    setCursor(0);
    setCaptureMode(false);
    setPlaying(false);
    setStatus("Ready");
    setFocus("");
  }
  function editValues(next: Record<string, any>) {
    if (events.length) resetInspection();
    setValues(next);
  }
  const visible = cursor === null ? events : events.slice(0, cursor + 1);
  const initialized = events.find((e) => e.type === "initialized");
  const last = visible.at(-1);
  const animationEvent = [...visible].reverse().find((e) => isPlaybackStep(e));
  const animationIndex = animationEvent ? events.indexOf(animationEvent) : -1;
  useEffect(() => {
    if (playing && followCamera && animationEvent)
      setFocus(eventFocus(animationEvent));
  }, [animationIndex, playing, followCamera]);
  const round =
    last?.round ?? [...visible].reverse().find((e) => e.round)?.round ?? 0;
  const diagnostics =
    [...visible].reverse().find((e) => e.type === "diagnostics")?.values ?? {};
  const decisionEvent = [...visible]
    .reverse()
    .find((e) =>
      ["screening_updated", "temporal_trust_updated", "diagnostics"].includes(
        e.type,
      ),
    );
  const isolatedClients =
    decisionEvent?.type === "diagnostics"
      ? (decisionEvent.values.isolated ?? [])
      : (decisionEvent?.isolated ?? []);
  const groups =
    [...visible].reverse().find((e) => e.type === "grouping_completed")
      ?.groups ??
    diagnostics.groups ??
    initialized?.groups ??
    {};
  const latestTraining = [...visible]
    .reverse()
    .find((e) => e.type.startsWith("client_training_"));
  const clients = initialized?.clients ?? [];
  const hierarchy = initialized
    ? Object.keys(initialized.groups).length > 0
    : method.startsWith("fedhep");
  const graph = useMemo(() => {
    if (!initialized) return { nodes: [] as Node[], edges: [] as Edge[] };
    const nodes: Node[] = [
      {
        id: "dataset",
        position: { x: 40, y: 0 },
        data: {
          label: (
            <>
              <strong>
                {initialized.dataset === "cifar100"
                  ? "CIFAR-100"
                  : initialized.dataset}{" "}
                partitions
              </strong>
              <span>
                {clients.reduce((n: number, c: any) => n + c.train, 0)} training
                examples
              </span>
            </>
          ),
        },
      },
      {
        id: "global",
        targetPosition: Position.Bottom,
        position: { x: 430, y: 0 },
        data: {
          label: (
            <>
              <strong>Global</strong>
              <span>{titles[initialized.method]}</span>
            </>
          ),
        },
      },
    ];
    const edges: Edge[] = [];
    const parentIds = (initialized.parents ??
      Array.from(new Set(Object.values(groups)))) as number[];
    parentIds
      .sort((a, b) => a - b)
      .forEach((g, index) =>
        nodes.push({
          id: `p${g}`,
          targetPosition: Position.Bottom,
          position: { x: 100 + index * 300, y: 170 },
          data: {
            label: (
              <>
                <strong>Parent {g}</strong>
                <span>Shared parent head</span>
              </>
            ),
          },
        }),
      );
    clients.forEach((c: any, index: number) => {
      const malicious = initialized.attackers.includes(c.id);
      const isolated = isolatedClients.includes(c.id);
      const excluded = (diagnostics.excluded ?? []).includes(c.id);
      const active =
        latestTraining?.client === c.id &&
        latestTraining.type === "client_training_started";
      nodes.push({
        id: `c${c.id}`,
        sourcePosition: Position.Top,
        position: {
          x: (index % 3) * 280 + 80,
          y: hierarchy
            ? 360 + Math.floor(index / 3) * 140
            : 200 + Math.floor(index / 3) * 140,
        },
        className: [
          malicious ? "malicious" : "honest",
          isolated || excluded ? "excluded" : "",
          active ? "active" : "",
        ].join(" "),
        data: {
          label: (
            <>
              <strong>Client {String(c.id).padStart(2, "0")}</strong>
              <span>
                {isolated
                  ? "Isolated"
                  : excluded
                    ? "Excluded"
                    : active
                      ? "Training"
                      : malicious
                        ? "Malicious"
                        : "Benign"}
              </span>
              <small>{c.train} training samples</small>
            </>
          ),
        },
      });
      edges.push({
        id: `data${c.id}`,
        source: "dataset",
        target: `c${c.id}`,
        hidden: selected !== `c${c.id}`,
        style: { stroke: "#666", strokeDasharray: "2 4" },
      });
      edges.push({
        id: `g${c.id}`,
        source: `c${c.id}`,
        target: "global",
        style: {
          stroke: "#888",
          strokeDasharray: excluded ? "5 5" : undefined,
        },
        markerEnd: { type: MarkerType.ArrowClosed, color: "#888" },
      });
      if (hierarchy)
        edges.push({
          id: `p${c.id}`,
          source: `c${c.id}`,
          target: `p${groups[c.id]}`,
          style: { stroke: "#aaa" },
          markerEnd: { type: MarkerType.ArrowClosed, color: "#aaa" },
        });
    });
    const transfers = edges.map((edge) => {
      const cid = Number(edge.source.slice(1));
      const event = animationEvent;
      const target =
        event?.tier === "parent" || event?.tier === "parent_defense"
          ? `p${event.parent}`
          : "global";
      const upload =
        event?.type === "upload_ready" &&
        event.tier === "global" &&
        event.client === cid &&
        edge.target === "global";
      const receive =
        event?.type === "aggregation_started" &&
        event.members?.includes(cid) &&
        edge.target === target;
      return {
        ...edge,
        type: "weight",
        data: {
          active: Boolean(upload || receive),
          playing,
          speed,
          sequence: animationIndex,
        },
      };
    });
    if (
      animationEvent &&
      ["Parents", "Global", "Screening"].includes(eventPhase(animationEvent))
    ) {
      const target =
        eventPhase(animationEvent) === "Parents"
          ? `p${animationEvent.parent}`
          : "global";
      const node = nodes.find((node) => node.id === target);
      if (node) node.className = "active";
    }
    return { nodes, edges: transfers };
  }, [
    initialized,
    groups,
    diagnostics,
    latestTraining,
    clients,
    hierarchy,
    selected,
    animationIndex,
    isolatedClients,
    playing,
    speed,
  ]);
  const metrics = visible
    .filter((e) => e.type === "evaluation_completed")
    .map((e) => ({
      round: e.round,
      accuracy: e.values.pooled,
      bottom10: e.values.bottom10,
    }));
  const clientId = selected.startsWith("c") ? Number(selected.slice(1)) : null;
  const selectedClient = clients.find((c: any) => c.id === clientId);
  const clientEvent = [...visible]
    .reverse()
    .find(
      (e) => e.client === clientId && e.type === "client_training_completed",
    );
  const lines = events
    .slice(clearAt)
    .filter((e) => e.type === "log" || e.type === "experiment_failed")
    .map(
      (e) =>
        `${new Date(e.timestamp * 1000).toLocaleTimeString()} ${e.type === "experiment_failed" ? "ERROR " : ""}${e.text ?? e.error}`,
    )
    .filter((l) => l.toLowerCase().includes(filter.toLowerCase()));
  const running = [
    "running",
    "starting",
    "paused",
    "cancelling",
    "queued",
    "pending",
  ].includes(status);
  const locked = running || busy || schemaLoading;
  const sections = Array.from(new Set(fields.map((f) => f.key.split(".")[0])));
  function displayField(f: Field) {
    const attack = values["attack.kind"];
    if (f.key.startsWith("attack.") && f.key !== "attack.kind" && !attack)
      return false;
    if (f.key.startsWith("attack.backdoor") || f.key === "attack.scale")
      return attack === "backdoor";
    if (["attack.knowledge", "attack.alie_z"].includes(f.key))
      return ["alie", "min_max"].includes(attack);
    return true;
  }
  return (
    <MotionConfig reducedMotion="user">
      <div className="app">
        <header>
          <div className="identity">
            RFL Lab<span>Research workbench</span>
          </div>
          <div className="run-title" title={capabilities.mode}>
            {titles[initialized?.method ?? method]}
            <span>
              {id
                ? `Visualization ${id.slice(0, 16)}`
                : "Choose a visualization"}
            </span>
          </div>
          <div className="run-status" role="status" aria-live="polite">
            <b>{status}</b>
            <span>
              Round {round} ·{" "}
              {(
                [...visible]
                  .reverse()
                  .find((e) => typeof e.elapsed_s === "number")?.elapsed_s ?? 0
              ).toFixed(1)}{" "}
              s
            </span>
          </div>
          <button onClick={() => setDark(!dark)} aria-label="Toggle theme">
            {dark ? "Light theme" : "Dark theme"}
          </button>
        </header>
        <nav aria-label="Main navigation">
          {[
            "Live Workbench",
            "Architecture",
            "Mathematical inspection",
            "Experiment history",
            "Documentation",
          ].map((item) => (
            <button
              key={item}
              aria-current={view === item ? "page" : undefined}
              onClick={() => navigate(item)}
            >
              {item}
            </button>
          ))}
        </nav>
        {error && (
          <div role="alert" className="error">
            {error}
            <button onClick={() => setError("")}>Dismiss</button>
          </div>
        )}
        {view === "Live Workbench" ? (
          <>
            <div className="sticky-workbench-controls">
              <button
                aria-expanded={!sidebarHidden}
                onClick={() => {
                  setSidebarHidden(!sidebarHidden);
                  localStorage.setItem(
                    "rfl-sidebar-hidden",
                    String(!sidebarHidden),
                  );
                }}
              >
                {sidebarHidden ? "Show settings" : "Hide settings"}
              </button>
              {initialized && (
                <div className="playback-controls">
                  <button
                    onClick={() => {
                      if (
                        !playing &&
                        !events.some(
                          (e, i) => i > (cursor ?? -1) && isPlaybackStep(e),
                        )
                      )
                        setCursor(
                          Math.max(
                            0,
                            events.findIndex((e) => e.type === "initialized"),
                          ),
                        );
                      setPlaying(!playing);
                    }}
                  >
                    {playing ? "Pause animation" : "Play animation"}
                  </button>
                  <button
                    disabled={
                      !events.some(
                        (e, i) => i < (cursor ?? 0) && isPlaybackStep(e),
                      )
                    }
                    onClick={() => {
                      const previous = events.findLastIndex(
                        (e, i) => i < (cursor ?? 0) && isPlaybackStep(e),
                      );
                      if (previous >= 0) seek(previous);
                    }}
                  >
                    Previous operation
                  </button>
                  <button
                    disabled={
                      !events.some(
                        (e, i) => i > (cursor ?? -1) && isPlaybackStep(e),
                      )
                    }
                    onClick={() => {
                      const next = events.findIndex(
                        (e, i) => i > (cursor ?? -1) && isPlaybackStep(e),
                      );
                      if (next >= 0) seek(next);
                    }}
                  >
                    Next operation
                  </button>
                  <label>
                    Animation speed
                    <select
                      aria-label="Animation speed"
                      value={speed}
                      onChange={(e) => setSpeed(Number(e.target.value))}
                    >
                      <option value={0.5}>0.5×</option>
                      <option value={1}>1×</option>
                      <option value={2}>2×</option>
                      <option value={4}>4×</option>
                    </select>
                  </label>
                  <label>
                    <input
                      type="checkbox"
                      checked={followCamera}
                      onChange={(e) => setFollowCamera(e.target.checked)}
                    />{" "}
                    Follow camera
                  </label>
                  <span>
                    {playing ? "Automatic playback" : "Paused"} ·{" "}
                    {
                      events.filter(
                        (e, i) => i > (cursor ?? -1) && isPlaybackStep(e),
                      ).length
                    }{" "}
                    operations available
                  </span>
                </div>
              )}
              {initialized && (
                <details className="sticky-trace">
                  <summary>Execution trace</summary>{" "}
                  <section className="timeline" aria-label="Execution timeline">
                    <div>
                      <h3>Execution trace</h3>
                      <button
                        onClick={() => {
                          setPlaying(false);
                          setCursor(Math.max(0, events.length - 1));
                        }}
                      >
                        Inspect latest available
                      </button>
                    </div>
                    <input
                      type="range"
                      aria-label="Execution event"
                      min={0}
                      max={Math.max(0, events.length - 1)}
                      value={cursor ?? Math.max(0, events.length - 1)}
                      onChange={(e) => seek(Number(e.target.value))}
                    />
                    <div className="event-strip">
                      {events
                        .filter(
                          (e) =>
                            ![
                              "log",
                              "upload_ready",
                              "operation_completed",
                              "optimizer_step",
                              "loss_backward_completed",
                            ].includes(e.type),
                        )
                        .slice(-100)
                        .map((e, i) => (
                          <button
                            key={i}
                            onClick={() => seek(events.indexOf(e))}
                            title={pretty(e)}
                          >
                            r{e.round ?? "—"} · {e.type.replaceAll("_", " ")}
                            {e.head ? ` · ${e.head} head` : ""}
                          </button>
                        ))}
                    </div>
                  </section>
                </details>
              )}
            </div>
            <main
              className={`workbench ${sidebarHidden ? "sidebar-hidden" : ""}`}
            >
              {!sidebarHidden && (
                <aside className="configuration">
                  <h2>Experiment setup</h2>
                  <p>Parameters are passed to the research implementation.</p>
                  {schemaLoading && (
                    <p role="status">Loading model settings…</p>
                  )}
                  <label>
                    Method
                    <select
                      aria-label="Method"
                      value={method}
                      disabled={locked}
                      onChange={(e) => {
                        resetInspection();
                        setMethod(e.target.value);
                        installConfiguration({
                          method: e.target.value,
                          overrides: {},
                        }).catch((error) => setError(error.message));
                      }}
                    >
                      {models.map((m) => (
                        <option key={m.id} value={m.id}>
                          {titles[m.id]}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label>
                    Visualization capture
                    <select
                      aria-label="Visualization capture"
                      value={captureId}
                      disabled={running}
                      onChange={(e) => setCaptureId(e.target.value)}
                    >
                      {captures.map((c) => (
                        <option key={c.id} value={c.id}>
                          {titles[c.method]} · {c.attack ?? "clean"}
                        </option>
                      ))}
                    </select>
                  </label>
                  <button
                    disabled={running || busy}
                    onClick={() => loadCapture()}
                  >
                    Load captured visualization
                  </button>
                  {(capabilities.requiresKey || !capabilities.training) && (
                    <details open={!cpuKey}>
                      <summary>
                        {capabilities.requiresKey
                          ? capabilities.mode
                          : "Connect your computer"}
                      </summary>
                      <label>
                        Computer bridge address
                        <input
                          type="url"
                          value={cpuEndpoint}
                          disabled={running}
                          placeholder="https://…trycloudflare.com"
                          onChange={(e) => setCpuEndpoint(e.target.value)}
                        />
                      </label>
                      <label>
                        Computer access key
                        <input
                          type="password"
                          autoComplete="off"
                          disabled={running}
                          value={cpuKey}
                          onChange={(e) => {
                            setCpuKey(e.target.value);
                            sessionStorage.setItem(
                              "rfl-cpu-key",
                              e.target.value,
                            );
                            localStorage.setItem("rfl-cpu-key", e.target.value);
                          }}
                        />
                      </label>
                      <button
                        disabled={running}
                        onClick={() => {
                          for (const storage of [
                            sessionStorage,
                            localStorage,
                          ]) {
                            if (cpuEndpoint.trim())
                              storage.setItem(
                                "rfl-cpu-url",
                                cpuEndpoint.trim(),
                              );
                            else storage.removeItem("rfl-cpu-url");
                          }
                          api("/capabilities")
                            .then(setCapabilities)
                            .catch((e) => setError(e.message));
                        }}
                      >
                        Connect computer
                      </button>
                      <button
                        disabled={running}
                        onClick={() => {
                          for (const storage of [
                            localStorage,
                            sessionStorage,
                          ]) {
                            storage.removeItem("rfl-cpu-key");
                            storage.removeItem("rfl-cpu-url");
                          }
                          setCpuKey("");
                          setCpuEndpoint("");
                          api("/capabilities").then(setCapabilities);
                        }}
                      >
                        Forget computer connection
                      </button>
                      <small>
                        The local bridge opens a connected browser tab. The key
                        is remembered in this browser.
                      </small>
                    </details>
                  )}
                  <div className="launch">
                    <label>
                      Training device
                      <select
                        aria-label="Training device"
                        disabled={locked}
                        value={values["run.device"] ?? "cpu"}
                        onChange={(event) =>
                          editValues({
                            ...values,
                            "run.device": event.target.value,
                          })
                        }
                      >
                        {(capabilities.devices ?? ["cpu"]).map((device) => (
                          <option key={device} value={device}>
                            {deviceLabels[device] ?? device}
                          </option>
                        ))}
                      </select>
                    </label>
                    <button
                      className="primary"
                      disabled={
                        busy ||
                        running ||
                        !fields.length ||
                        !capabilities.training ||
                        !capabilities.online ||
                        (capabilities.requiresKey && !cpuKey)
                      }
                      onClick={start}
                    >
                      {busy ? "Validating…" : "Run visualization"}
                    </button>
                    {(captureMode || (initialized && !running)) &&
                      capabilities.training && (
                        <button onClick={resetInspection}>
                          Configure a new visualization
                        </button>
                      )}
                    {status === "paused" && !captureMode && (
                      <button
                        onClick={() =>
                          api(`/experiments/${id}/resume`, {})
                            .then((run) => setStatus(run.status))
                            .catch((error) => setError(error.message))
                        }
                      >
                        Resume execution
                      </button>
                    )}
                    {running && (
                      <button
                        onClick={() =>
                          api(`/experiments/${id}/cancel`, {})
                            .then(() => setStatus("cancelling"))
                            .catch((e) => setError(e.message))
                        }
                      >
                        Cancel execution
                      </button>
                    )}
                    <small>
                      {captureMode
                        ? "Editing a setting starts a new configuration. Run visualization to compute its results."
                        : capabilities.training
                          ? "Run these settings on your connected computer using the selected device. Results wait for Next operation."
                          : "Choose a captured execution to inspect its actual computations."}
                    </small>
                  </div>
                  {sections.map((section) => (
                    <details
                      key={section}
                      open={["data", "train", "run"].includes(section)}
                    >
                      <summary>
                        {section === "plus" ? "FedHEP+ screening" : section}
                      </summary>
                      {fields
                        .filter(
                          (f) =>
                            f.key.startsWith(section + ".") &&
                            f.key !== "run.device" &&
                            displayField(f),
                        )
                        .map((f) => (
                          <label key={f.key}>
                            {f.key
                              .split(".")
                              .slice(1)
                              .join(" ")
                              .replaceAll("_", " ")}
                            {f.options ? (
                              <select
                                disabled={locked}
                                value={
                                  values[f.key] === null
                                    ? "null"
                                    : String(values[f.key])
                                }
                                onChange={(e) =>
                                  editValues({
                                    ...values,
                                    [f.key]:
                                      e.target.value === "null"
                                        ? null
                                        : e.target.value,
                                  })
                                }
                              >
                                {f.options
                                  .filter(
                                    (v) =>
                                      f.key !== "run.device" ||
                                      (
                                        capabilities.devices ?? ["cpu"]
                                      ).includes(v),
                                  )
                                  .map((v) => (
                                    <option key={String(v)} value={String(v)}>
                                      {f.key === "run.device"
                                        ? deviceLabels[v]
                                        : v === null
                                          ? "Disabled"
                                          : v === "synthetic"
                                            ? "Synthetic (test fixture)"
                                            : v}
                                    </option>
                                  ))}
                              </select>
                            ) : f.type === "boolean" ? (
                              <input
                                type="checkbox"
                                disabled={locked}
                                checked={!!values[f.key]}
                                onChange={(e) =>
                                  editValues({
                                    ...values,
                                    [f.key]: e.target.checked,
                                  })
                                }
                              />
                            ) : (
                              <input
                                disabled={locked}
                                type={f.type === "number" ? "number" : "text"}
                                step="any"
                                placeholder={
                                  f.key === "data.alpha"
                                    ? "Blank = IID"
                                    : undefined
                                }
                                value={values[f.key] ?? ""}
                                onChange={(e) =>
                                  editValues({
                                    ...values,
                                    [f.key]:
                                      e.target.value === ""
                                        ? null
                                        : f.type === "number"
                                          ? Number(e.target.value)
                                          : e.target.value,
                                  })
                                }
                              />
                            )}
                          </label>
                        ))}
                    </details>
                  ))}
                </aside>
              )}
              <section className="center">
                <div className="canvas-heading">
                  <div>
                    <h1>Live topology</h1>
                    <p>
                      {initialized
                        ? "Select a component to inspect its recorded state."
                        : "Load a visualization to reveal its actual initialization."}
                    </p>
                  </div>
                  <span>
                    {clients.length
                      ? `${clients.length} clients`
                      : busy ||
                          (!captureMode &&
                            id &&
                            ["running", "pending", "queued"].includes(status))
                        ? "Awaiting initialization"
                        : "Ready"}
                  </span>
                </div>
                <div className="canvas">
                  {initialized ? (
                    <ReactFlow
                      key={id}
                      edgeTypes={edgeTypes}
                      nodes={graph.nodes}
                      edges={graph.edges}
                      fitView
                      fitViewOptions={{ padding: 0.2 }}
                      minZoom={0.15}
                      zoomOnScroll={false}
                      panOnScroll={false}
                      preventScrolling={false}
                      onNodeClick={(_, node) => setSelected(node.id)}
                      nodesDraggable={false}
                      colorMode={dark ? "dark" : "light"}
                    >
                      <AutoFit focus={focus} />
                      <Background color={dark ? "#303030" : "#ddd"} gap={22} />
                      <Controls />
                      <MiniMap
                        nodeColor="#888"
                        maskColor={dark ? "#111111bb" : "#ffffffbb"}
                      />
                    </ReactFlow>
                  ) : (
                    <div className="empty">
                      <h2>Inspect a real execution.</h2>
                      <p>
                        Choose a model and attack capture, then load the
                        visualization. Play its training operations, weight
                        transfers and server decisions. Local Python execution
                        is also available on your workstation.
                      </p>
                      <span>ResNet-9 · Real execution · Persisted events</span>
                    </div>
                  )}
                </div>
              </section>
              <motion.aside
                className="inspector"
                key={selected}
                initial={{ opacity: 0.6 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.15 }}
              >
                <h2>
                  {selectedClient
                    ? `Client ${clientId}`
                    : selected === "global"
                      ? "Global inspection"
                      : `Parent ${selected.slice(1)}`}
                </h2>
                <p>Values captured by the execution adapter</p>
                {selectedClient ? (
                  <>
                    <ClientDistribution
                      client={selectedClient}
                      malicious={initialized?.attackers.includes(clientId)}
                      parent={groups[clientId!]}
                    />
                    <h3>Latest local training</h3>
                    <pre>
                      {clientEvent
                        ? pretty(clientEvent)
                        : "No completed training captured."}
                    </pre>
                    <details>
                      <summary>Captured local operations</summary>
                      {visible
                        .filter(
                          (e) =>
                            e.client === clientId &&
                            [
                              "operation_completed",
                              "optimizer_step",
                              "loss_backward_completed",
                            ].includes(e.type),
                        )
                        .slice(-90)
                        .map((e, i) => (
                          <details key={i}>
                            <summary>
                              {e.operation ?? "Optimizer step"} {e.module ?? ""}
                            </summary>
                            <pre>{pretty(e)}</pre>
                          </details>
                        ))}
                    </details>
                    <h3>Screening record</h3>
                    <pre>
                      {pretty({
                        score: diagnostics.scores?.[clientId!],
                        z: diagnostics.z?.[clientId!],
                        evidence: diagnostics.evidence?.[clientId!],
                      })}
                    </pre>
                  </>
                ) : (
                  <>
                    {selected.startsWith("p") && (
                      <>
                        <h3>Parent aggregation</h3>
                        <pre>
                          {pretty(
                            [...visible]
                              .reverse()
                              .find(
                                (e) =>
                                  e.type === "aggregation_completed" &&
                                  e.parent === Number(selected.slice(1)),
                              ) ?? "No aggregation recorded.",
                          )}
                        </pre>
                      </>
                    )}
                    <h3>Round diagnostics</h3>
                    <pre>
                      {Object.keys(diagnostics).length
                        ? pretty(diagnostics)
                        : "No round diagnostics captured."}
                    </pre>
                  </>
                )}
                <details>
                  <summary>Actual module architecture</summary>
                  <pre>
                    {initialized?.architecture ??
                      "Available after initialization."}
                  </pre>
                </details>
                <details>
                  <summary>Selected execution event</summary>
                  <pre>{last ? pretty(last) : "No event selected."}</pre>
                </details>
              </motion.aside>
              <section className="execution-progress">
                <label htmlFor="round-progress">
                  Completed rounds ·{" "}
                  {events.filter((e) => e.type === "round_completed").length}/
                  {initialized?.configured_rounds ?? values["run.rounds"] ?? 1}
                </label>
                <progress
                  id="round-progress"
                  max={
                    initialized?.configured_rounds ?? values["run.rounds"] ?? 1
                  }
                  value={
                    events.filter((e) => e.type === "round_completed").length
                  }
                />
              </section>
              <ExecutionAnimation
                event={animationEvent}
                sequence={animationIndex}
                initialized={initialized}
                formulas={formulas}
              />
              <RoundWalkthrough
                events={visible}
                currentEvent={animationEvent}
                initialized={initialized}
                formulas={formulas}
                jump={(phase: string) => {
                  const index = events.findIndex(
                    (e) =>
                      e.round === (round || 1) &&
                      eventPhase(e) === phase &&
                      e.type !== "initialized",
                  );
                  if (index >= 0) seek(index);
                }}
                focus={(id) => {
                  setSelected(id);
                  setFocus(id);
                }}
              />
              <section className="metrics" aria-label="Evaluation metrics">
                <div className="metrics-heading">
                  <h3>Evaluation</h3>
                  <span>Accuracy % · all evaluated clients</span>
                </div>
                {metrics.length ? (
                  <ResponsiveContainer width="100%" height={240}>
                    <LineChart data={metrics}>
                      <CartesianGrid
                        stroke={dark ? "#333" : "#ddd"}
                        vertical={false}
                      />
                      <XAxis dataKey="round" stroke="#888" />
                      <YAxis domain={[0, 100]} stroke="#888" />
                      <Tooltip
                        contentStyle={{
                          background: dark ? "#222" : "#fff",
                          border: "1px solid #888",
                          color: dark ? "#fff" : "#111",
                        }}
                      />
                      <Line
                        name="Pooled accuracy"
                        dataKey="accuracy"
                        stroke={dark ? "#fff" : "#111"}
                        isAnimationActive={false}
                      />
                      <Line
                        name="Bottom 10%"
                        dataKey="bottom10"
                        stroke="#888"
                        strokeDasharray="4 4"
                        isAnimationActive={false}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                ) : (
                  <p>
                    Metrics appear when the configured evaluation completes.
                  </p>
                )}
              </section>
            </main>
          </>
        ) : (
          <section className="document-view">
            {view === "Mathematical inspection" ? (
              <>
                <h1>Mathematical inspection</h1>
                <p>
                  Implementation explanations. Captured runtime values are
                  available in the component inspector and event trace.
                </p>
                {formulas
                  .filter(
                    (f) =>
                      f.method === "all" ||
                      f.method === method ||
                      (f.method === "fedhep" && method.startsWith("fedhep")) ||
                      (f.method === "fedavg" &&
                        [
                          "fedavg",
                          "fedprox",
                          "scaffold",
                          "ditto",
                          "fedrep",
                          "fedhep",
                        ].includes(method)),
                  )
                  .map((f) => (
                    <article key={f.name}>
                      <h2>{f.name}</h2>
                      <Markdown
                        remarkPlugins={[remarkMath]}
                        rehypePlugins={[rehypeKatex]}
                      >{`$$\n${f.latex}\n$$\n\n${f.explanation}`}</Markdown>
                      <button
                        onClick={() =>
                          fetch("/api/source/" + f.source)
                            .then((r) => r.text())
                            .then(setSource)
                        }
                      >
                        Read {f.source}
                      </button>
                    </article>
                  ))}
                {source && (
                  <details open>
                    <summary>Research source</summary>
                    <pre>{source}</pre>
                  </details>
                )}
              </>
            ) : view === "Experiment history" ? (
              <ExperimentHistory
                targetId={historyTarget}
                history={history}
                refresh={() =>
                  api("/history?refresh=true")
                    .then(setHistory)
                    .catch((e) => setError(e.message))
                }
                loadConfig={async (config: any) => {
                  if (running) {
                    setError(
                      "Wait for the current visualization to finish or cancel it before loading another configuration.",
                    );
                    return;
                  }
                  resetInspection();
                  try {
                    await installConfiguration({
                      method: config.method,
                      overrides: { ...config.overrides, "run.device": "cpu" },
                    });
                  } catch (error) {
                    setError((error as Error).message);
                  }
                  navigate("Live Workbench");
                }}
                replay={async (run: any) => {
                  await installConfiguration(run.config);
                  setStatus(run.status);
                  connect(run.id);
                  navigate("Live Workbench");
                }}
                cancel={(runId: string) =>
                  api(`/experiments/${runId}/cancel`, {})
                    .then(() => api("/history"))
                    .then(setHistory)
                    .catch((e) => setError(e.message))
                }
              />
            ) : view === "Architecture" ? (
              <>
                <h1>Execution architecture</h1>
                <p>
                  All methods run the original Python implementations. Consensus
                  methods have a Global server and clients. FedHEP methods
                  additionally maintain parent heads.
                </p>
                <h2>{titles[method]}</h2>
                <p>
                  {method === "fedhep_robust"
                    ? "Local training → Global robust rule → parent SCCF → evaluation"
                    : method === "fedhep_plus"
                      ? "Local training → screening and evidence → optional regrouping → parent SCCF → Global aggregation → evaluation"
                      : method === "fedhep_defended"
                        ? "Local training → parent SCCF and trust → Global SCCF and trust → evaluation"
                        : method === "fedhep"
                          ? "Local training → grouping → parent update and momentum → Global weighted mean → evaluation"
                          : "Local training → aggregation → Global update → evaluation"}
                </p>
                <h2>Communication semantics</h2>
                <p>
                  The research executes locally using tensors. Client uploads go
                  directly to Global and, in the FedHEP family, to their parent.
                  Parent models supply the parent head on the next local
                  training pass. Transfer sizes represent tensor payloads.
                  Network latency is unavailable.
                </p>
                <h2>Observed ResNet-9 operations</h2>
                <p>
                  First captured forward per module for client 0, round 1. The
                  order records executed calls. Residual additions are defined
                  in the source architecture.
                </p>
                <div className="operation-sequence">
                  {events
                    .filter(
                      (e) =>
                        e.type === "operation_completed" &&
                        e.operation === "forward" &&
                        e.client === 0 &&
                        e.round === 1,
                    )
                    .map((e, i) => (
                      <details key={i}>
                        <summary>
                          {e.module} · {e.module_type}
                        </summary>
                        <p>
                          Output shape {pretty(e.shape)} · mean {e.output_mean}
                        </p>
                        <pre>{pretty(e)}</pre>
                      </details>
                    ))}
                </div>
                <h2>Instantiated module tree</h2>
                <pre>
                  {initialized?.architecture ??
                    "Run an experiment to inspect the instantiated module tree."}
                </pre>
              </>
            ) : (
              <Documentation />
            )}
          </section>
        )}
        <footer>
          <div className="terminal-toolbar">
            <button onClick={() => setTerminalOpen(!terminalOpen)}>
              {terminalOpen ? "Collapse" : "Expand"} process output
            </button>
            <input
              aria-label="Filter process output"
              placeholder="Filter output…"
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
            />
            <button
              onClick={() =>
                navigator.clipboard
                  .writeText(lines.join("\n"))
                  .catch((e) => setError(e.message))
              }
            >
              Copy
            </button>
            <button onClick={() => setClearAt(events.length)}>
              Clear view
            </button>
            <span>{events.length} recorded events</span>
          </div>
          {terminalOpen && <LogTerminal lines={lines} dark={dark} />}
        </footer>
      </div>
    </MotionConfig>
  );
}
