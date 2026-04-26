"""Tests for API endpoints."""
import pytest
from io import BytesIO
from PIL import Image
import numpy as np

class TestHealthCheck:
    """Test root health check endpoint."""
    
    def test_read_root(self, client):
        """Test GET / returns correct status."""
        test_client, _ = client
        response = test_client.get("/")
        assert response.status_code == 200
        assert response.json() == {"status": "service functioning"}


class TestPredictEndpoint:
    """Test /predict endpoint."""
    
    def create_dummy_image(self):
        """Helper to create a dummy image file."""
        img = Image.new('RGB', (224, 224), color = 'red')
        img_byte_arr = BytesIO()
        img.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)
        return img_byte_arr

    def test_predict_image(self, client, monkeypatch):
        """Test prediction for an uploaded image."""
        test_client, mock_session = client
        
        # Mock the ONNX session.run output
        # We expect a list of outputs, each a numpy array
        # Set class 3 (cat) to have highest value
        mock_output = np.zeros((1, 10), dtype=np.float32)
        mock_output[0, 3] = 10.0 
        
        # Mock session.run(None, inputs)
        mock_session.run.return_value = [mock_output]
        
        # Create a dummy image
        img_bytes = self.create_dummy_image()
        
        # Upload using multipart
        files = {"file": ("test.jpg", img_bytes, "image/jpeg")}
        response = test_client.post("/predict", files=files)
        
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] == "cat"
        assert "confidence" in data
        assert "latency_ms" in data

    def test_predict_invalid_file_type(self, client):
        """Test prediction with invalid file type."""
        test_client, _ = client
        
        files = {"file": ("test.txt", BytesIO(b"Hello world"), "text/plain")}
        response = test_client.post("/predict", files=files)
        
        assert response.status_code == 400
        assert "not an image" in response.json()["detail"].lower()
