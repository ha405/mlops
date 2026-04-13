import pytest
from fastapi.testclient import TestClient
from app.serve import app
import joblib
from unittest.mock import patch, MagicMock


@pytest.fixture
def client():
    """Create a test client for the API."""
    return TestClient(app)


@pytest.fixture
def mock_model():
    """Mock the iris model."""
    model = MagicMock()
    # setosa = 0, versicolor = 1, virginica = 2
    model.predict.return_value = [0]
    return model


def test_read_root(client):
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "service functioning"}


def test_predict_endpoint(client, mock_model):
    """Test the prediction endpoint."""
    with patch.object(client.app.state, 'model', mock_model):
        response = client.post(
            "/predict",
            json={
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }
        )
    
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "latency_ms" in data
    assert data["prediction"] == "setosa"


def test_predict_versicolor(client, mock_model):
    """Test prediction for versicolor iris."""
    mock_model.predict.return_value = [1]
    with patch.object(client.app.state, 'model', mock_model):
        response = client.post(
            "/predict",
            json={
                "sepal_length": 6.0,
                "sepal_width": 2.7,
                "petal_length": 5.1,
                "petal_width": 1.6
            }
        )
    
    assert response.status_code == 200
    assert response.json()["prediction"] == "versicolor"


def test_predict_virginica(client, mock_model):
    """Test prediction for virginica iris."""
    mock_model.predict.return_value = [2]
    with patch.object(client.app.state, 'model', mock_model):
        response = client.post(
            "/predict",
            json={
                "sepal_length": 7.0,
                "sepal_width": 3.2,
                "petal_length": 6.0,
                "petal_width": 2.0
            }
        )
    
    assert response.status_code == 200
    assert response.json()["prediction"] == "virginica"


def test_predict_invalid_data(client):
    """Test prediction with invalid data."""
    response = client.post(
        "/predict",
        json={
            "sepal_length": "invalid",
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
    )
    assert response.status_code == 422  # Validation error
