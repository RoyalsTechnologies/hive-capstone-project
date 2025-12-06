"""
Tests for health check endpoints
"""

from fastapi.testclient import TestClient

from app.main import app
from app.services.model_service import ModelService


def test_health_check():
    """Test basic health check endpoint"""
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data


def test_readiness_check_models_loaded():
    """Test readiness check when models are loaded"""
    # Ensure models are loaded
    if not ModelService.are_models_loaded():
        ModelService.load_models()

    client = TestClient(app)
    response = client.get("/health/ready")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["models_loaded"] is True
    assert "timestamp" in data


def test_readiness_check_models_not_loaded():
    """Test readiness check when models are not loaded"""
    # Unload models
    ModelService.unload_models()

    client = TestClient(app)
    response = client.get("/health/ready")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "not ready"
    assert data["models_loaded"] is False

    # Reload models for other tests
    ModelService.load_models()
