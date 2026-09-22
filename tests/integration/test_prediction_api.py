from fastapi.testclient import (
    TestClient,
)

from backend.app.main import app


client = TestClient(app)


ARTICLE = """
Researchers published a detailed report after conducting
multiple experiments. Independent researchers reviewed the
results and recommended additional studies to confirm the
findings.
"""


def test_health_endpoint():

    response = client.get(
        "/health"
    )

    assert response.status_code == 200

    data = response.json()

    assert "status" in data
    assert "model_ready" in data


def test_root_endpoint():

    response = client.get("/")

    assert response.status_code == 200


def test_model_info_endpoint():

    response = client.get(
        "/model-info"
    )

    assert response.status_code == 200

    data = response.json()

    assert "model" in data
    assert "feature_type" in data


def test_prediction_endpoint():

    response = client.post(
        "/predict",
        json={
            "text": ARTICLE
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] in [
        "FAKE",
        "REAL",
    ]

    assert data["label_id"] in [
        0,
        1,
    ]

    assert "decision_score" in data

    assert "request_id" in data


def test_prediction_validation():

    response = client.post(
        "/predict",
        json={
            "text": "Short"
        },
    )

    assert response.status_code == 422
