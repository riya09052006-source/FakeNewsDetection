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


def test_history_and_analytics_flow():

    # 1. Clean history to start with a known baseline
    client.delete("/history")

    # 2. Check analytics is empty
    analytics_before = client.get("/analytics").json()
    assert analytics_before["total"] == 0

    # 3. Post a prediction
    pred_res = client.post(
        "/predict",
        json={
            "text": (
                "Researchers published a detailed scientific "
                "report after conducting multiple experiments."
            )
        },
    )
    assert pred_res.status_code == 200

    # 4. Check history contains 1 item
    hist_res = client.get("/history")
    assert hist_res.status_code == 200
    hist_data = hist_res.json()
    assert hist_data["count"] >= 1
    first_id = hist_data["items"][0]["id"]

    # 5. Check detail endpoint
    detail_res = client.get(f"/history/{first_id}")
    assert detail_res.status_code == 200
    detail_data = detail_res.json()
    assert detail_data["id"] == first_id
    assert "text" in detail_data

    # 6. Check analytics updated
    analytics_after = client.get("/analytics").json()
    assert analytics_after["total"] >= 1

    # 7. Clear history
    clear_res = client.delete("/history")
    assert clear_res.status_code == 200
    assert clear_res.json()["deleted_count"] >= 1
