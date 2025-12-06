"""
Tests for model service
"""

import numpy as np
import pytest

from app.services.model_service import DummyModel, ModelService


class TestModelService:
    """Test cases for ModelService"""

    def setup_method(self):
        """Setup before each test method"""
        ModelService.unload_models()
        ModelService.load_models()

    def teardown_method(self):
        """Cleanup after each test method"""
        ModelService.unload_models()

    def test_load_models(self):
        """Test that models can be loaded"""
        ModelService.unload_models()
        assert not ModelService.are_models_loaded()

        ModelService.load_models()
        assert ModelService.are_models_loaded()

    def test_unload_models(self):
        """Test that models can be unloaded"""
        ModelService.load_models()
        assert ModelService.are_models_loaded()

        ModelService.unload_models()
        assert not ModelService.are_models_loaded()

    def test_are_models_loaded(self):
        """Test model loading status check"""
        ModelService.unload_models()
        assert ModelService.are_models_loaded() is False

        ModelService.load_models()
        assert ModelService.are_models_loaded() is True

    def test_predict_single(self):
        """Test single prediction"""
        features = [1.0, 2.0, 3.0, 4.0]
        prediction, confidence = ModelService.predict(features)

        assert prediction is not None
        assert isinstance(prediction, (int, float, list))
        # Confidence may be None if model doesn't support it
        assert confidence is None or isinstance(confidence, float)

    def test_predict_batch(self):
        """Test batch prediction"""
        features_list = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        predictions = ModelService.predict_batch(features_list)

        assert len(predictions) == len(features_list)
        for pred, conf in predictions:
            assert pred is not None
            assert isinstance(pred, (int, float, list))
            assert conf is None or isinstance(conf, float)

    def test_predict_invalid_model(self):
        """Test prediction with invalid model name"""
        with pytest.raises(ValueError, match="not found"):
            ModelService.predict([1.0, 2.0], model_name="nonexistent")

    def test_predict_batch_invalid_model(self):
        """Test batch prediction with invalid model name"""
        with pytest.raises(ValueError, match="not found"):
            ModelService.predict_batch([[1.0, 2.0]], model_name="nonexistent")

    def test_list_models(self):
        """Test listing available models"""
        models = ModelService.list_models()
        assert isinstance(models, list)
        assert "default" in models


class TestDummyModel:
    """Test cases for DummyModel"""

    def test_dummy_model_predict(self):
        """Test dummy model prediction"""
        model = DummyModel()
        X = np.array([[1.0, 2.0, 3.0]])
        predictions = model.predict(X)

        assert len(predictions) == 1
        assert predictions[0] == 6.0  # Sum of features

    def test_dummy_model_predict_proba(self):
        """Test dummy model probability prediction"""
        model = DummyModel()
        X = np.array([[1.0, 2.0, 3.0]])
        proba = model.predict_proba(X)

        assert proba.shape == (1, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)  # Probabilities sum to 1
