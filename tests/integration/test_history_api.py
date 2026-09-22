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


def test_prediction_creates_history():

    response = client.post(
        "/predict",
        json={
            "text": ARTICLE
        },
    )

    assert response.status_code == 200

    history_response = client.get(
        "/history"
    )

    assert history_response.status_code == 200

    data = history_response.json()

    assert "count" in data
    assert "items" in data

    assert data["count"] >= 1


def test_analytics_endpoint():

    response = client.get(
        "/analytics"
    )

    assert response.status_code == 200

    data = response.json()

    assert "total" in data
    assert "fake_count" in data
    assert "real_count" in data

    assert "fake_percentage" in data
    assert "real_percentage" in data


def test_history_detail():

    prediction_response = client.post(
        "/predict",
        json={
            "text": ARTICLE
        },
    )

    prediction_response.raise_for_status()

    history_response = client.get(
        "/history"
    )

    data = history_response.json()

    assert data["count"] >= 1

    prediction_id = data["items"][0]["id"]

    detail_response = client.get(
        f"/history/{prediction_id}"
    )

    assert detail_response.status_code == 200

    detail = detail_response.json()

    assert "text" in detail
    assert "prediction" in detail


def test_missing_history_item():

    response = client.get(
        "/history/999999999"
    )

    assert response.status_code == 404
