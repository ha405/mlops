import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient


@pytest.fixture
def mock_model():
    """Create a mock sklearn-like model."""
    model = MagicMock()
    # Mock sklearn estimator behavior
    model.predict = MagicMock(return_value=[0])  # Default: setosa
    return model


@pytest.fixture
def app_with_mock_model(mock_model, monkeypatch):
    """Create FastAPI app with mocked model, avoiding file I/O."""
    # Mock joblib.load to return our mock model
    def mock_load(path):
        return mock_model
    
    monkeypatch.setattr('joblib.load', mock_load)
    
    # Import app AFTER patching joblib
    from app.serve import app
    return app, mock_model


@pytest.fixture
def client(app_with_mock_model):
    """Create TestClient with mocked model."""
    app, mock_model = app_with_mock_model
    return TestClient(app), mock_model


