from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health() -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_analyze_endpoint() -> None:
    r = client.post(
        "/analyze",
        json={"comments": [{"id": "1", "text": "thank you!"}, {"id": "2", "text": "I will kill you"}]},
    )
    assert r.status_code == 200
    data = r.json()
    assert len(data["results"]) == 2