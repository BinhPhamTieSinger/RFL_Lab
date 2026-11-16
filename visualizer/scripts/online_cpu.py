#!/usr/bin/env python3
"""Connect the hosted workbench to an authenticated, CPU-only local worker."""

import argparse
import fcntl
import json
import os
import re
import secrets
import signal
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path
from urllib.parse import urlencode

VIZ = Path(__file__).resolve().parents[1]
STATE = VIZ / "data/online-cpu"
SITE = "https://rfl-lab-visualizer.netlify.app"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-open", action="store_true", help="Keep the connected browser tab closed")
    args = parser.parse_args()
    STATE.mkdir(parents=True, exist_ok=True)
    bridge_lock = (STATE / "bridge.lock").open("a")
    try:
        fcntl.flock(bridge_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        connection_file = STATE / "connection.json"
        if connection_file.exists() and not args.no_open:
            endpoint = json.loads(connection_file.read_text())["endpoint"]
            token = (STATE / "access-key").read_text().strip()
            webbrowser.open(f"{SITE}/#{urlencode({'cpu-api': endpoint, 'cpu-key': token})}")
        print("The CPU bridge is already running.")
        return
    (STATE / "connection.json").unlink(missing_ok=True)
    binary = VIZ / "data/bin/cloudflared"
    if not binary.is_file():
        sys.exit("Install cloudflared in visualizer/data/bin/cloudflared as described in docs/ONLINE_CPU.md.")
    token_file = STATE / "access-key"
    if not token_file.exists():
        with open(token_file, "x", opener=lambda p, flags: os.open(p, flags, 0o600)) as handle:
            handle.write(secrets.token_urlsafe(32))
    token = token_file.read_text().strip()
    env = {
        **os.environ,
        "RFL_VIZ_DATA": str(STATE / "runs"),
        "RFL_VIZ_LOCK_PATH": str(VIZ / "data/training.lock"),
        "RFL_VIZ_TOKEN": token,
        "RFL_VIZ_CPU_ONLY": "1",
        "PYTHONPATH": f"{VIZ / 'backend'}{os.pathsep}{VIZ.parent / 'src'}",
    }
    children = []

    def stop(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    with (STATE / "api.log").open("a") as api_log, (STATE / "tunnel.log").open("w+") as tunnel_log:
        try:
            api = subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "rfl_viz.api:app", "--host", "127.0.0.1", "--port", "8001"],
                env=env,
                stdout=api_log,
                stderr=subprocess.STDOUT,
            )
            children.append(api)
            for _ in range(60):
                if api.poll() is not None:
                    sys.exit("CPU API could not start. Inspect data/online-cpu/api.log.")
                try:
                    urllib.request.urlopen("http://127.0.0.1:8001/health", timeout=1).close()
                    break
                except OSError:
                    time.sleep(1)
            else:
                sys.exit("CPU API did not become ready.")
            tunnel = subprocess.Popen(
                [
                    str(binary),
                    "tunnel",
                    "--url",
                    "http://127.0.0.1:8001",
                    "--protocol",
                    "http2",
                    "--no-prechecks",
                    "--edge-ip-version",
                    "4",
                    "--no-autoupdate",
                ],
                stdout=tunnel_log,
                stderr=subprocess.STDOUT,
            )
            children.append(tunnel)
            endpoint = None
            retries = 0
            for _ in range(90):
                if tunnel.poll() is not None:
                    if retries >= 3:
                        sys.exit("Tunnel could not start after retries. Inspect data/online-cpu/tunnel.log.")
                    retries += 1
                    time.sleep(2)
                    tunnel_log.seek(0)
                    tunnel_log.truncate()
                    tunnel = subprocess.Popen(tunnel.args, stdout=tunnel_log, stderr=subprocess.STDOUT)
                    children.append(tunnel)
                    continue
                connection_log = (STATE / "tunnel.log").read_text()
                match = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", connection_log)
                if match and "Registered tunnel connection" in connection_log:
                    endpoint = match.group(0)
                    break
                time.sleep(1)
            else:
                sys.exit("Tunnel did not become reachable.")
            (STATE / "connection.json").write_text(
                json.dumps({"endpoint": endpoint, "site": SITE, "pid": os.getpid()})
            )
            print(
                "CPU bridge online. Keep this process running. The bridge opens a connected workbench tab.",
                flush=True,
            )
            if not args.no_open:
                webbrowser.open(f"{SITE}/#{urlencode({'cpu-api': endpoint, 'cpu-key': token})}")
            while api.poll() is None and tunnel.poll() is None:
                time.sleep(1)
            sys.exit("CPU bridge process exited. Inspect data/online-cpu logs and restart this script.")
        except KeyboardInterrupt:
            print("Stopping CPU bridge.", flush=True)
        finally:
            for child in reversed(children):
                if child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()


if __name__ == "__main__":
    main()
