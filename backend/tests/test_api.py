from fastapi.testclient import TestClient

from backend.app.main import app


client = TestClient(app)


def test_root():

    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "running"


def test_health():

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert "status" in data

    assert "model_ready" in data


def test_model_info():

    response = client.get("/model-info")

    assert response.status_code == 200

    data = response.json()

    assert "model" in data

    assert "feature_type" in data

    assert "prediction_labels" in data


def test_prediction():

    payload = {
        "text": (
            "Researchers published a detailed scientific "
            "report after conducting multiple experiments. "
            "The researchers stated that additional studies "
            "are necessary to confirm the findings."
        )
    }

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] in {
        "FAKE",
        "REAL",
    }

    assert "decision_score" in data

    assert "request_id" in data


def test_empty_prediction():

    payload = {
        "text": ""
    }

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422


def test_batch_prediction():

    payload = {
        "texts": [
            (
                "Researchers published a detailed scientific "
                "report after conducting multiple experiments "
                "and stated that additional studies are needed."
            ),
            (
                "A viral post claims that a secret drink can "
                "instantly cure every disease without providing "
                "scientific evidence."
            ),
        ]
    }

    response = client.post(
        "/predict/batch",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["count"] == 2

    assert len(data["results"]) == 2


def test_explain_endpoint():

    response = client.post(
        "/explain",
        json={
            "text": (
                "A viral social media post claims that a secret "
                "drink can instantly cure every known disease. "
                "The post provides no scientific evidence."
            )
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert "decision_score" in data
    assert "top_real_features" in data
    assert "top_fake_features" in data
    assert "explanation_note" in data
    assert "request_id" in data
