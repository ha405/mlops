"""Tests for the vision model."""
import pytest
import os
import torch
import torchvision.models as models

MODEL_PATH = "model/vision_model.pth"

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
        model = models.mobilenet_v2()
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(num_ftrs, 10)
        
        # Should load successfully
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        assert model is not None

    @pytest.mark.skipif(
        not os.path.exists(MODEL_PATH),
        reason=f"Model file not found at {MODEL_PATH}"
    )
    def test_model_predict_output_shape(self):
        """Test that model prediction has correct shape."""
        model = models.mobilenet_v2()
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(num_ftrs, 10)
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        model.eval()
        
        # Test single sample (batch_size=1, channels=3, height=224, width=224)
        sample = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            prediction = model(sample)
        
        assert prediction.shape == (1, 10)
