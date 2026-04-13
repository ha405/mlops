"""Tests for API endpoints."""
import pytest


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
    
    def test_predict_setosa(self, client):
        """Test prediction for setosa iris."""
        test_client, mock_model = client
        mock_model.predict.return_value = [0]
        
        response = test_client.post(
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
        assert data["prediction"] == "setosa"
        assert "latency_ms" in data
        assert data["latency_ms"] >= 0

    def test_predict_versicolor(self, client):
        """Test prediction for versicolor iris."""
        test_client, mock_model = client
        mock_model.predict.return_value = [1]
        
        response = test_client.post(
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

    def test_predict_virginica(self, client):
        """Test prediction for virginica iris."""
        test_client, mock_model = client
        mock_model.predict.return_value = [2]
        
        response = test_client.post(
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

    def test_predict_missing_field(self, client):
        """Test prediction with missing required field."""
        test_client, _ = client
        
        response = test_client.post(
            "/predict",
            json={
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                # Missing petal_length and petal_width
            }
        )
        
        assert response.status_code == 422  # Validation error

    def test_predict_invalid_type(self, client):
        """Test prediction with invalid data type."""
        test_client, _ = client
        
        response = test_client.post(
            "/predict",
            json={
                "sepal_length": "not_a_number",
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }
        )
        
        assert response.status_code == 422  # Validation error

    def test_model_called_with_correct_features(self, client):
        """Test that model.predict is called with correct features."""
        test_client, mock_model = client
        mock_model.predict.return_value = [1]
        
        test_client.post(
            "/predict",
            json={
                "sepal_length": 5.5,
                "sepal_width": 3.0,
                "petal_length": 1.5,
                "petal_width": 0.3
            }
        )
        
        # Verify model was called
        assert mock_model.predict.called
        # Get the features passed to predict
        call_args = mock_model.predict.call_args
        features = call_args[0][0]  # First positional argument
        assert features == [[5.5, 3.0, 1.5, 0.3]]

