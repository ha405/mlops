import pytest
from unittest.mock import MagicMock, patch
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(autouse=True)
def patch_joblib_load(monkeypatch):
    """Patch joblib.load for all tests to avoid loading actual model files."""
    mock_model = MagicMock()
    mock_model.predict.return_value = [0]
    monkeypatch.setattr('joblib.load', MagicMock(return_value=mock_model))
    return mock_model


@pytest.fixture
def mock_model():
    """Create a fresh mock model for each test."""
    model = MagicMock()
    model.predict.return_value = [0]
    return model

