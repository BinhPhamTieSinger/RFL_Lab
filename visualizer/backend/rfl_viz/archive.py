"""Index saved research outputs without rewriting or inferring their results."""

import hashlib
import json
import threading
import time

import yaml

from rfl_viz.catalog import LAB

_lock = threading.Lock()
_cached = (0.0, {})


def flatten(config):
    return {
        f"{section}.{key}": value
        for section, values in config.items()
        if isinstance(values, dict)
        for key, value in values.items()
    }


def index(refresh=False):
    global _cached
    with _lock:
        if not refresh and time.monotonic() - _cached[0] < 900:
            return _cached[1]
        entries = {}
        for root in sorted(LAB.glob("outputs*")):
            if not root.is_dir():
                continue
            files = set(root.rglob("metrics.json")) | set(root.rglob("config.yaml"))
            for folder in sorted({p.parent for p in files}):
                relative = folder.relative_to(LAB).as_posix()
                id = "archive-" + hashlib.sha256(relative.encode()).hexdigest()[:24]
                metrics = folder / "metrics.json"
                config_path = folder / "config.yaml"
                try:
                    config = yaml.safe_load(config_path.read_text()) if config_path.exists() else {}
                    result = json.loads(metrics.read_text()) if metrics.exists() else None
                    config = config or {}
                    method = config.get("method", (result or {}).get("method", "unknown"))
                    entries[id] = {
                        "id": id,
                        "status": "completed" if result and result.get("final") else "incomplete",
                        "created": (metrics if metrics.exists() else config_path).stat().st_mtime,
                        "date_basis": "file modification time",
                        "name": config.get("name", folder.name),
                        "source": relative,
                        "archived": True,
                        "config": {"method": method, "overrides": flatten(config)},
                        "final": (result or {}).get("final"),
                        "folder": folder,
                    }
                except (OSError, ValueError, yaml.YAMLError) as error:
                    entries[id] = {
                        "id": id,
                        "status": "unreadable",
                        "created": folder.stat().st_mtime,
                        "name": folder.name,
                        "source": relative,
                        "archived": True,
                        "config": {"method": "unknown", "overrides": {}},
                        "error": str(error),
                        "folder": folder,
                    }
        _cached = (time.monotonic(), entries)
        return entries


def history(refresh=False):
    return [{k: v for k, v in item.items() if k != "folder"} for item in index(refresh).values()]


def detail(id):
    item = index().get(id)
    if item is None:
        return None
    folder = item["folder"]
    result_path = folder / "metrics.json"
    result = None
    try:
        result = json.loads(result_path.read_text()) if result_path.exists() else None
    except (OSError, ValueError):
        pass
    logs = []
    for path in sorted(folder.glob("*.log")):
        logs.append({"name": path.name, "text": path.read_text(errors="replace")[-100_000:]})
    return {k: v for k, v in item.items() if k != "folder"} | {"result": result, "logs": logs}
