from fastapi.testclient import (
    TestClient,
)

from backend.app.main import app


client = TestClient(app)


ARTICLE = """
A viral social media post claims that a secret drink
can instantly cure every known disease. The post provides
no scientific evidence or credible source to support
the claim.
"""


def test_explain_endpoint():

    response = client.post(
        "/explain",
        json={
            "text": ARTICLE
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert "label_id" in data
    assert "decision_score" in data

    assert "top_real_features" in data
    assert "top_fake_features" in data

    assert "explanation_note" in data

    assert "request_id" in data


def test_explain_validation():

    response = client.post(
        "/explain",
        json={
            "text": "Too short"
        },
    )

    assert response.status_code == 422
