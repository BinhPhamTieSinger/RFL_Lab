"use client";
import { useEffect, useRef } from "react";
import { Terminal } from "@xterm/xterm";
import { FitAddon } from "@xterm/addon-fit";
export default function LogTerminal({
  lines,
  dark,
}: {
  lines: string[];
  dark: boolean;
}) {
  const host = useRef<HTMLDivElement>(null);
  const terminal = useRef<Terminal | null>(null);
  useEffect(() => {
    const t = new Terminal({
      fontSize: 12,
      fontFamily: "Menlo, monospace",
      convertEol: true,
      disableStdin: true,
      scrollback: 10000,
    });
    const fit = new FitAddon();
    t.loadAddon(fit);
    t.open(host.current!);
    terminal.current = t;
    const resize = new ResizeObserver(() => {
      try {
        fit.fit();
      } catch {}
    });
    resize.observe(host.current!);
    return () => {
      resize.disconnect();
      terminal.current = null;
      t.dispose();
    };
  }, []);
  useEffect(() => {
    const t = terminal.current;
    if (!t) return;
    t.options.theme = {
      background: dark ? "#111111" : "#f6f6f6",
      foreground: dark ? "#dddddd" : "#222222",
      selectionBackground: dark ? "#555555" : "#cccccc",
    };
    t.reset();
    for (const line of lines)
      t.writeln(line.replace(/\x1b\[[0-9;]*[a-zA-Z]/g, ""));
  }, [lines, dark]);
  return (
    <div
      ref={host}
      className="terminal-surface"
      aria-label="Actual worker log"
    />
  );
}
