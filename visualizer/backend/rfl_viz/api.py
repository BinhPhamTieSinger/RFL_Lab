"""Local-only API with isolated training workers, SQLite replay and a process-wide GPU lock."""

import asyncio
import fcntl
import json
import os
import secrets
import signal
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager, suppress
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from rfl_viz import archive
from rfl_viz.catalog import FORMULAS, LAB, METHODS, schema, validate

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("RFL_VIZ_DATA", ROOT / "data"))
DATA.mkdir(parents=True, exist_ok=True)


def db():
    con = sqlite3.connect(DATA / "experiments.sqlite", timeout=30)
    con.row_factory = sqlite3.Row
    return con


with db() as con:
    con.executescript(
        "CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY, status TEXT, created REAL, config TEXT);"
        "CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY AUTOINCREMENT, run TEXT, payload TEXT);"
        "CREATE INDEX IF NOT EXISTS events_run ON events(run,seq);"
    )
processes = {}
mutex = threading.Lock()


def push(id, payload):
    payload = {"v": 1, "experiment": id, "timestamp": time.time(), **payload}
    with db() as con:
        con.execute("INSERT INTO events(run,payload) VALUES(?,?)", (id, json.dumps(payload)))


def status(id, value):
    with db() as con:
        con.execute("UPDATE runs SET status=? WHERE id=?", (value, id))


def supervise(id, process, lock):
    outcome = "failed"
    try:
        for line in process.stdout:
            if line.startswith("@rfl "):
                event = json.loads(line[5:])
                push(id, event)
                if event["type"] == "experiment_completed":
                    outcome = "completed"
            else:
                push(id, {"type": "log", "text": line.rstrip()})
        code = process.wait()
        with db() as con:
            row = con.execute("SELECT status FROM runs WHERE id=?", (id,)).fetchone()
        if row["status"] == "cancelling":
            outcome = "cancelled"
        elif code:
            outcome = "failed"
        status(id, outcome)
        push(id, {"type": "worker_stopped", "status": outcome, "exit_code": code})
    finally:
        with mutex:
            processes.pop(id, None)
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


class RunRequest(BaseModel):
    method: str = "fedhep_plus"
    overrides: dict = Field(default_factory=dict)


@asynccontextmanager
async def lifespan(app):
    # A worker's output pipe closes if the API exits. Recover interrupted metadata on restart.
    with db() as con:
        con.execute("UPDATE runs SET status='interrupted' WHERE status IN ('running','paused','cancelling')")
    stop = threading.Event()

    def schedule():
        while not stop.wait(0.5):
            dispatch()

    scheduler = threading.Thread(target=schedule, daemon=True)
    scheduler.start()
    yield
    stop.set()
    scheduler.join(timeout=2)
    for process in list(processes.values()):
        if process.poll() is None:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGCONT)
                os.killpg(process.pid, signal.SIGTERM)


app = FastAPI(title="RFL Lab Visualizer", lifespan=lifespan)


@app.middleware("http")
async def local_requests(request: Request, call_next):
    token = os.environ.get("RFL_VIZ_TOKEN")
    if token and request.url.path != "/health" and not secrets.compare_digest(
        request.headers.get("authorization", "").encode(), f"Bearer {token}".encode()
    ):
        from fastapi.responses import JSONResponse

        return JSONResponse({"detail": "Connect your CPU using the local bridge access key."}, status_code=401)
    origin = request.headers.get("origin")
    if (
        request.method in ("POST", "PUT", "DELETE")
        and origin
        and origin not in ("http://localhost:3000", "http://127.0.0.1:3000")
    ):
        from fastapi.responses import JSONResponse

        return JSONResponse({"detail": "Untrusted origin"}, status_code=403)
    return await call_next(request)


@lru_cache(maxsize=1)
def detected_devices():
    import torch
    devices = ["cpu"]
    for name, backend in (("mps", torch.backends.mps), ("cuda", torch.cuda), ("xpu", getattr(torch, "xpu", None))):
        if backend is not None and backend.is_available():
            devices.append(name)
    return devices


def available_devices():
    if os.environ.get("RFL_VIZ_CPU_ONLY") == "1":
        return ["cpu"]
    return [*detected_devices(), "auto"]


@app.get("/health")
def health():
    return {"online": True, "device": "cpu" if os.environ.get("RFL_VIZ_CPU_ONLY") == "1" else "configured", "devices": available_devices()}


@app.get("/models")
def models():
    return [{"id": m, "hierarchical": m.startswith("fedhep")} for m in METHODS]


@app.get("/models/{method}/schema")
def model_schema(method: str):
    try:
        return schema(method)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e


@app.get("/formulas")
def formulas():
    return FORMULAS


@app.post("/experiments/validate")
def check(body: RunRequest):
    if os.environ.get("RFL_VIZ_CPU_ONLY") == "1" and body.overrides.get("run.device", "cpu") != "cpu":
        raise HTTPException(422, "The online computer bridge uses CPU. Select cpu for run.device.")
    device = body.overrides.get("run.device", "cpu")
    if device not in available_devices():
        raise HTTPException(422, f"Device {device} is unavailable on the connected computer. Available devices: {', '.join(available_devices())}")
    try:
        validate(body.method, body.overrides)
    except (ValueError, TypeError) as e:
        raise HTTPException(422, str(e)) from e
    return {"valid": True}


def dispatch():
    with mutex:
        if processes:
            return
        lock = Path(os.environ.get("RFL_VIZ_LOCK_PATH", DATA / "training.lock")).open("a")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close()
            return
        with db() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute("SELECT id FROM runs WHERE status='queued' ORDER BY created LIMIT 1").fetchone()
            if row is None:
                lock.close()
                return
            id = row["id"]
            con.execute("UPDATE runs SET status='running' WHERE id=?", (id,))
        env = {
            **os.environ,
            "PYTHONPATH": str(ROOT / "backend") + os.pathsep + str(LAB / "src"),
            "OMP_NUM_THREADS": "2",
        }
        try:
            process = subprocess.Popen(
                [sys.executable, "-u", "-m", "rfl_viz.worker", str(DATA / id)],
                cwd=LAB,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                pass_fds=(lock.fileno(),),
                start_new_session=True,
            )
        except Exception as error:
            status(id, "failed")
            push(id, {"type": "worker_stopped", "status": "failed", "error": str(error)})
            lock.close()
            return
        processes[id] = process
        threading.Thread(target=supervise, args=(id, process, lock), daemon=True).start()


@app.post("/experiments")
def launch(body: RunRequest):
    check(body)
    id = uuid.uuid4().hex
    folder = DATA / id
    folder.mkdir()
    raw = body.model_dump()
    (folder / "request.json").write_text(json.dumps(raw))
    with db() as con:
        con.execute("INSERT INTO runs VALUES(?,?,?,?)", (id, "pending", time.time(), json.dumps(raw)))
    push(id, {"type": "experiment_pending"})
    push(id, {"type": "experiment_queued"})
    status(id, "queued")
    dispatch()
    return {"id": id, "status": detail(id)["status"]}


@app.get("/experiments")
def history(refresh: bool = False):
    with db() as con:
        return [
            dict(r) | {"config": json.loads(r["config"])}
            for r in con.execute("SELECT * FROM runs ORDER BY created DESC")
        ] + archive.history(refresh)


@app.get("/experiments/{id}")
def detail(id: str):
    if id.startswith("archive-"):
        item = archive.detail(id)
        if item is None:
            raise HTTPException(404, "Unknown archived experiment")
        return item
    with db() as con:
        row = con.execute("SELECT * FROM runs WHERE id=?", (id,)).fetchone()
    if row is None:
        raise HTTPException(404, "Unknown experiment")
    path = DATA / id / "result.json"
    return dict(row) | {
        "config": json.loads(row["config"]),
        "result": json.loads(path.read_text()) if path.exists() else None,
    }


@app.post("/experiments/{id}/pause")
def pause(id: str):
    detail(id)
    with mutex:
        process = processes.get(id)
        if process and process.poll() is None:
            with db() as con:
                changed = con.execute(
                    "UPDATE runs SET status='paused' WHERE id=? AND status='running'", (id,)
                ).rowcount
            if changed:
                try:
                    os.killpg(process.pid, signal.SIGSTOP)
                except ProcessLookupError:
                    pass
                push(id, {"type": "execution_paused"})
        else:
            with db() as con:
                con.execute("UPDATE runs SET status='paused' WHERE id=? AND status IN ('queued','pending')", (id,))
    return {"status": detail(id)["status"]}


@app.post("/experiments/{id}/resume")
def resume(id: str):
    detail(id)
    with mutex:
        if detail(id)["status"] == "paused":
            process = processes.get(id)
            if process and process.poll() is None:
                status(id, "running")
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGCONT)
                push(id, {"type": "execution_resumed"})
            elif process is None:
                status(id, "queued")
    dispatch()
    return {"status": detail(id)["status"]}


@app.post("/experiments/{id}/cancel")
def cancel(id: str):
    run = detail(id)
    if run.get("archived"):
        raise HTTPException(409, "Archived runs are read-only")
    with mutex:
        with db() as con:
            changed = con.execute(
                "UPDATE runs SET status='cancelled' WHERE id=? AND status IN ('pending','queued')", (id,)
            ).rowcount
        if not changed and id not in processes:
            with db() as con:
                changed = con.execute("UPDATE runs SET status='cancelled' WHERE id=? AND status='paused'", (id,)).rowcount
        if changed:
            push(id, {"type": "worker_stopped", "status": "cancelled"})
        process = processes.get(id)
        if process and process.poll() is None:
            status(id, "cancelling")
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGCONT)
                os.killpg(process.pid, signal.SIGTERM)

            def kill_later():
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    with suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)

            threading.Thread(target=kill_later, daemon=True).start()
    return {"status": detail(id)["status"]}


@app.get("/experiments/{id}/events")
async def events(id: str, request: Request, after: int = 0):
    detail(id)
    cursor = max(after, int(request.headers.get("last-event-id", "0")))

    async def stream():
        nonlocal cursor
        while not await request.is_disconnected():
            with db() as con:
                rows = con.execute(
                    "SELECT seq,payload FROM events WHERE run=? AND seq>? ORDER BY seq LIMIT 500", (id, cursor)
                ).fetchall()
            for row in rows:
                cursor = row["seq"]
                yield f"id: {cursor}\ndata: {row['payload']}\n\n"
            if not rows:
                if detail(id)["status"] not in ("running", "paused", "cancelling", "queued", "pending"):
                    break
                yield ": heartbeat\n\n"
                await asyncio.sleep(0.25)

    return StreamingResponse(
        stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


@app.get("/experiments/{id}/event-batch")
def event_batch(id: str, after: int = 0):
    run = detail(id)
    with db() as con:
        rows = con.execute(
            "SELECT seq,payload FROM events WHERE run=? AND seq>? ORDER BY seq LIMIT 100", (id, max(0, after))
        ).fetchall()
    return {
        "events": [json.loads(row["payload"]) for row in rows],
        "cursor": rows[-1]["seq"] if rows else after,
        "status": run["status"],
        "more": len(rows) == 100,
    }


@app.get("/source/{path:path}", response_class=PlainTextResponse)
def source(path: str):
    file = (LAB / "src/rfl" / path).resolve()
    if not file.is_relative_to((LAB / "src/rfl").resolve()) or file.suffix != ".py" or not file.is_file():
        raise HTTPException(404)
    return file.read_text()


@app.get("/documentation", response_class=PlainTextResponse)
def documentation():
    return (ROOT / "README.md").read_text()
