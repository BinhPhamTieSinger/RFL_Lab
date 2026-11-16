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

VIZ = Path(__file__).resolve().parents[1]
STATE = VIZ / "data/online-cpu"
SITE = "https://rfl-lab-visualizer.netlify.app"
SITE_ID = "e8c6b306-0467-4ca4-8c41-0c31e1bfb675"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-deploy", action="store_true", help="Update the endpoint without triggering a new build"
    )
    parser.add_argument("--no-open", action="store_true", help="Keep the connected browser tab closed")
    args = parser.parse_args()
    STATE.mkdir(parents=True, exist_ok=True)
    bridge_lock = (STATE / "bridge.lock").open("a")
    try:
        fcntl.flock(bridge_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit("The CPU bridge is already running.")
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
            for _ in range(90):
                if tunnel.poll() is not None:
                    sys.exit("Tunnel could not start. Inspect data/online-cpu/tunnel.log.")
                match = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", (STATE / "tunnel.log").read_text())
                if match:
                    endpoint = match.group(0)
                    try:
                        urllib.request.urlopen(endpoint + "/health", timeout=4).close()
                        break
                    except OSError:
                        pass
                time.sleep(1)
            else:
                sys.exit("Tunnel did not become reachable.")
            (STATE / "connection.json").write_text(
                json.dumps({"endpoint": endpoint, "site": SITE, "pid": os.getpid()})
            )
            cli = ["npx", "--yes", "netlify-cli"]
            subprocess.run(
                cli
                + [
                    "env:set",
                    "RFL_BACKEND_URL",
                    endpoint,
                    "--context",
                    "production",
                    "--scope",
                    "functions",
                    "--force",
                ],
                cwd=VIZ / "frontend",
                check=True,
                stdout=subprocess.DEVNULL,
            )
            if not args.no_deploy:
                subprocess.run(
                    cli + ["api", "createSiteBuild", "--data", json.dumps({"site_id": SITE_ID})],
                    cwd=VIZ / "frontend",
                    check=True,
                    stdout=subprocess.DEVNULL,
                )
            print(
                "CPU bridge online. Keep this process running. "
                "Netlify must finish deployment to use the new endpoint.",
                flush=True,
            )
            if not args.no_open:
                webbrowser.open(f"{SITE}/#cpu-key={token}")
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
