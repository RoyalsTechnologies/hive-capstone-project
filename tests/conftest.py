"""
Pytest configuration and shared fixtures
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.model_service import ModelService


@pytest.fixture(scope="function")
def client():
    """Create a test client for the FastAPI app"""
    # Ensure models are loaded for testing
    if not ModelService.are_models_loaded():
        ModelService.load_models()

    with TestClient(app) as test_client:
        yield test_client

    # Cleanup after each test
    ModelService.unload_models()


@pytest.fixture(scope="function")
def sample_features():
    """Sample feature vector for testing"""
    return [1.0, 2.0, 3.0, 4.0]


@pytest.fixture(scope="function")
def sample_batch_features():
    """Sample batch feature vectors for testing"""
    return [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
