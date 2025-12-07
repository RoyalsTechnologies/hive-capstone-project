"""
Tests for inference endpoints
"""

import pytest

from app.services.model_service import ModelService


@pytest.fixture(autouse=True)
def setup_models():
    """Ensure models are loaded before each test"""
    if not ModelService.are_models_loaded():
        ModelService.load_models()
    yield
    # Cleanup is handled by conftest


def test_single_prediction(client, sample_features):
    """Test single prediction endpoint"""
    response = client.post("/api/v1/predict", json={"features": sample_features})

    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert isinstance(data["prediction"], (int, float, list))


def test_single_prediction_empty_features(client):
    """Test single prediction with empty features"""
    response = client.post("/api/v1/predict", json={"features": []})

    # Empty features should be rejected for security (422 = validation error)
    assert response.status_code in [400, 422]


def test_batch_prediction(client, sample_batch_features):
    """Test batch prediction endpoint"""
    response = client.post("/api/v1/predict/batch", json={"features": sample_batch_features})

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == len(sample_batch_features)

    for item in data:
        assert "prediction" in item


def test_batch_prediction_single_item(client):
    """Test batch prediction with single item"""
    response = client.post("/api/v1/predict/batch", json={"features": [[1.0, 2.0, 3.0]]})

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 1


def test_prediction_request_validation(client):
    """Test that invalid request data is rejected"""
    # Missing features
    response = client.post("/api/v1/predict", json={})
    assert response.status_code == 422

    # Invalid feature type
    response = client.post("/api/v1/predict", json={"features": "not a list"})
    assert response.status_code == 422


def test_batch_prediction_request_validation(client):
    """Test that invalid batch request data is rejected"""
    # Missing features
    response = client.post("/api/v1/predict/batch", json={})
    assert response.status_code == 422

    # Invalid feature type
    response = client.post("/api/v1/predict/batch", json={"features": "not a list"})
    assert response.status_code == 422
