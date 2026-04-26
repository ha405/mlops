"""Tests for the vision model (ONNX)."""
import pytest
import os
import numpy as np
import onnxruntime as ort

MODEL_PATH = "model/vision_model.onnx"

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
        """Test that ONNX model file can be loaded."""
        session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
        assert session is not None

    @pytest.mark.skipif(
        not os.path.exists(MODEL_PATH),
        reason=f"Model file not found at {MODEL_PATH}"
    )
    def test_model_predict_output_shape(self):
        """Test that model prediction has correct shape."""
        session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
        
        # Test single sample (batch_size=1, channels=3, height=224, width=224)
        sample = np.random.randn(1, 3, 224, 224).astype(np.float32)
        inputs = {session.get_inputs()[0].name: sample}
        outputs = session.run(None, inputs)
        
        assert outputs[0].shape == (1, 10)
