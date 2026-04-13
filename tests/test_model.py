import pytest
import joblib
import os
from pathlib import Path


@pytest.fixture
def model_path():
    """Path to the iris model."""
    return "model/iris_model.pkl"


def test_model_exists(model_path):
    """Test that the model file exists."""
    assert os.path.exists(model_path), f"Model file not found at {model_path}"


def test_model_can_be_loaded(model_path):
    """Test that the model can be loaded."""
    model = joblib.load(model_path)
    assert model is not None, "Failed to load model"


def test_model_has_predict_method(model_path):
    """Test that the model has a predict method."""
    model = joblib.load(model_path)
    assert hasattr(model, 'predict'), "Model does not have predict method"


def test_model_prediction_shape(model_path):
    """Test that model predictions have expected shape."""
    model = joblib.load(model_path)
    
    # Test with a single sample
    sample = [[5.1, 3.5, 1.4, 0.2]]
    prediction = model.predict(sample)
    
    assert prediction.shape == (1,), f"Expected shape (1,), got {prediction.shape}"
    assert prediction[0] in [0, 1, 2], "Prediction should be 0, 1, or 2"


def test_model_prediction_range(model_path):
    """Test that predictions are in valid range."""
    model = joblib.load(model_path)
    
    # Test multiple samples
    samples = [
        [5.1, 3.5, 1.4, 0.2],  # setosa
        [6.0, 2.7, 5.1, 1.6],  # versicolor
        [7.0, 3.2, 6.0, 2.0]   # virginica
    ]
    
    for sample in samples:
        prediction = model.predict([sample])
        assert 0 <= prediction[0] <= 2, f"Prediction {prediction[0]} out of range [0, 2]"
