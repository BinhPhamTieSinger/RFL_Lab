import os
import tempfile
import time

os.environ["RFL_VIZ_DATA"] = tempfile.mkdtemp(prefix="rfl-viz-test-")
from fastapi.testclient import TestClient
from rfl_viz.api import app


def test_execution_replay_queue_and_cancellation():
    with TestClient(app) as client:
        assert len(client.get("/models").json()) == 10
        assert client.post("/experiments", json={"overrides": {"run.rounds": 0}}).status_code == 422
        response = client.post(
            "/experiments",
            json={
                "method": "fedhep_plus",
                "overrides": {"data.dataset": "synthetic", "data.train_subset": 120, "data.test_subset": 60},
            },
        )
        assert response.status_code == 200
        id = response.json()["id"]
        queued = client.post("/experiments", json={}).json()
        assert queued["status"] == "queued"
        assert client.post(f"/experiments/{queued['id']}/cancel").json()["status"] == "cancelled"
        deadline = time.monotonic() + 90
        while client.get("/experiments/" + id).json()["status"] == "running" and time.monotonic() < deadline:
            time.sleep(0.2)
        result = client.get("/experiments/" + id).json()
        assert result["status"] == "completed", result
        events = client.get(f"/experiments/{id}/events").text
        assert "evaluation_completed" in events and "worker_stopped" in events
        assert result["result"]["final"]["pooled"] >= 0
        response = client.post("/experiments", json={"overrides": {"run.rounds": 100}})
        id = response.json()["id"]
        client.post(f"/experiments/{id}/cancel")
        deadline = time.monotonic() + 15
        while client.get("/experiments/" + id).json()["status"] == "cancelling" and time.monotonic() < deadline:
            time.sleep(0.1)
        assert client.get("/experiments/" + id).json()["status"] == "cancelled"
        assert client.get("/experiments/not-a-run").status_code == 404
        assert (
            client.post("/experiments", json={}, headers={"origin": "https://foreign.example"}).status_code == 403
        )
