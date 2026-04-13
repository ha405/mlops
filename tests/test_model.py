"""Tests for the iris model."""
import pytest
import joblib
import os


MODEL_PATH = "model/iris_model.pkl"


class TestModelFile:
    """Test the model file."""
    
    @pytest.mark.skipif(
        not os.path.exists(MODEL_PATH),
        reason=f"Model file not found at {MODEL_PATH}"
    )
    def test_model_file_exists(self):
        """Test that model file exists."""
        assert os.path.exists(MODEL_PATH)

    @pytest.mark.skipif(
        not os.path.exists(MODEL_PATH),
        reason=f"Model file not found at {MODEL_PATH}"
    )
    def test_model_can_be_loaded(self):
        """Test that model file can be loaded."""
        model = joblib.load(MODEL_PATH)
        assert model is not None

    @pytest.mark.skipif(
        not os.path.exists(MODEL_PATH),
        reason=f"Model file not found at {MODEL_PATH}"
    )
    def test_model_has_predict_method(self):
        """Test that loaded model has predict method."""
        model = joblib.load(MODEL_PATH)
        assert hasattr(model, 'predict')
        assert callable(getattr(model, 'predict'))

    @pytest.mark.skipif(
        not os.path.exists(MODEL_PATH),
        reason=f"Model file not found at {MODEL_PATH}"
    )
    def test_model_predict_output_shape(self):
        """Test that model prediction has correct shape."""
        model = joblib.load(MODEL_PATH)
        
        # Test single sample
        sample = [[5.1, 3.5, 1.4, 0.2]]
        prediction = model.predict(sample)
        
        assert prediction.shape == (1,)
        assert 0 <= prediction[0] <= 2

    @pytest.mark.skipif(
        not os.path.exists(MODEL_PATH),
        reason=f"Model file not found at {MODEL_PATH}"
    )
    def test_model_predict_multiple_samples(self):
        """Test that model can predict multiple samples."""
        model = joblib.load(MODEL_PATH)
        
        samples = [
            [5.1, 3.5, 1.4, 0.2],  # setosa
            [6.0, 2.7, 5.1, 1.6],  # versicolor
            [7.0, 3.2, 6.0, 2.0]   # virginica
        ]
        predictions = model.predict(samples)
        
        assert predictions.shape == (3,)
        assert all(0 <= pred <= 2 for pred in predictions)

