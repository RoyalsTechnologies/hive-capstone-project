"""
Tests for main application
"""

from fastapi.testclient import TestClient

from app.main import app


def test_root_endpoint():
    """Test the root endpoint redirects to home"""
    client = TestClient(app)
    response = client.get("/", follow_redirects=False)

    # Root endpoint should redirect to /home
    assert response.status_code == 307  # Temporary redirect
    assert response.headers["location"] == "/home"


def test_api_docs_available():
    """Test that API documentation is available"""
    client = TestClient(app)
    response = client.get("/docs")
    assert response.status_code == 200


def test_openapi_schema():
    """Test that OpenAPI schema is available"""
    client = TestClient(app)
    response = client.get("/openapi.json")

    assert response.status_code == 200
    schema = response.json()
    assert schema["info"]["title"] == "Hive Weather Inference API"
    assert schema["info"]["version"] == "1.0.0"
