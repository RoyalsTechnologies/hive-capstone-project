"""
Tests for model service
"""

import numpy as np
import pytest

from app.services.model_service import ModelService


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

