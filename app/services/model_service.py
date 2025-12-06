"""
Model service for loading and managing weather inference models
"""

import logging
import os
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)


class ModelService:
    """Service for managing weather inference models"""

    _models: Dict[str, Any] = {}
    _models_loaded: bool = False

    @classmethod
    def _validate_model_path(cls, file_path: str) -> bool:
        """
        Validate that the model path is safe and within the allowed directory.
        Prevents path traversal attacks.

        Args:
            file_path: Path to validate

        Returns:
            True if path is safe, False otherwise
        """
        try:
            # Resolve to absolute path
            resolved_path = Path(file_path).resolve()
            # Get the base model directory as absolute path
            base_path = Path(settings.MODEL_PATH).resolve()

            # Check if resolved path is within base path
            # This prevents path traversal attacks (e.g., ../../../etc/passwd)
            return str(resolved_path).startswith(str(base_path))
        except Exception:
            return False

    @classmethod
    def load_models(cls):
        """Load weather inference models from disk"""
        try:
            # Load models for each team member
            for member_name in settings.TEAM_MEMBERS:
                # Try to load model file for this member
                # Format: models/{member_name_sanitized}.pkl
                sanitized_name = member_name.lower().replace(" ", "_")
                # Remove any potentially dangerous characters
                sanitized_name = "".join(
                    c for c in sanitized_name if c.isalnum() or c in ("_", "-")
                )
                model_path = os.path.join(settings.MODEL_PATH, f"{sanitized_name}.pkl")

                # Validate path before using
                if not cls._validate_model_path(model_path):
                    logger.warning(
                        f"Invalid model path for {member_name}: {model_path}. "
                        f"Using dummy model."
                    )
                    cls._models[member_name] = DummyModel()
                    continue

                if os.path.exists(model_path):
                    try:
                        # SECURITY NOTE: pickle.load() can execute arbitrary code.
                        # In production, consider using joblib or validating model files.
                        # For now, we only load from trusted sources (models/ directory).
                        with open(model_path, "rb") as f:
                            cls._models[member_name] = pickle.load(f)
                        logger.info(f"Model loaded for {member_name} from {model_path}")
                    except Exception as e:
                        logger.warning(
                            f"Error loading model for {member_name}: {str(e)}. "
                            f"Using dummy model."
                        )
                        cls._models[member_name] = DummyModel()
                else:
                    # Use dummy model if file doesn't exist
                    logger.info(
                        f"Model file not found for {member_name} at {model_path}. "
                        f"Using dummy model."
                    )
                    cls._models[member_name] = DummyModel()

            # Also load default model if it exists
            default_model_path = os.path.join(settings.MODEL_PATH, settings.MODEL_NAME)
            if cls._validate_model_path(default_model_path) and os.path.exists(default_model_path):
                try:
                    with open(default_model_path, "rb") as f:
                        cls._models["default"] = pickle.load(f)
                    logger.info(f"Default model loaded from {default_model_path}")
                except Exception as e:
                    logger.warning(f"Error loading default model: {str(e)}. Using dummy model.")
                    if "default" not in cls._models:
                        cls._models["default"] = DummyModel()
            # Always ensure default model exists for backward compatibility
            if "default" not in cls._models:
                cls._models["default"] = DummyModel()
                logger.info("Using default dummy model.")

            cls._models_loaded = True
            logger.info(f"Models loaded successfully. Available models: {list(cls._models.keys())}")
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            # Fallback to dummy model
            if not cls._models:
                cls._models["default"] = DummyModel()
            cls._models_loaded = True

    @classmethod
    def unload_models(cls):
        """Unload models from memory"""
        cls._models.clear()
        cls._models_loaded = False
        logger.info("Models unloaded")

    @classmethod
    def are_models_loaded(cls) -> bool:
        """Check if models are loaded"""
        return cls._models_loaded and len(cls._models) > 0

    @classmethod
    def predict(
        cls, features: List[float], model_name: str = "default"
    ) -> Tuple[Any, Optional[float]]:
        """
        Make a prediction using the specified model

        Args:
            features: List of feature values
            model_name: Name of the model to use

        Returns:
            Tuple of (prediction, confidence)

        Raises:
            ValueError: If model not found or input validation fails
        """
        # Input validation
        if not features:
            raise ValueError("Features list cannot be empty")
        if len(features) > 1000:  # Reasonable limit to prevent DoS
            raise ValueError("Feature list too long (max 1000 features)")
        if not all(isinstance(f, (int, float)) for f in features):
            raise ValueError("All features must be numeric")
        # Check for NaN or Inf values
        if any(not np.isfinite(f) for f in features):
            raise ValueError("Features must be finite numbers")

        # Validate model name to prevent injection
        # Allow alphanumeric, spaces, underscores, and hyphens (for team member names)
        if not isinstance(model_name, str) or not re.match(r"^[a-zA-Z0-9_\s-]+$", model_name):
            raise ValueError("Invalid model name")

        if model_name not in cls._models:
            raise ValueError(
                f"Model '{model_name}' not found. Available models: {list(cls._models.keys())}"
            )

        model = cls._models[model_name]

        # Convert to numpy array
        features_array = np.array(features).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features_array)

        # Try to get prediction probabilities for confidence
        confidence = None
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(features_array)
                confidence = float(np.max(proba))
            except Exception:
                pass

        # Convert numpy types to Python native types
        if isinstance(prediction, np.ndarray):
            prediction = prediction.tolist()[0]
        else:
            prediction = float(prediction)

        return prediction, confidence

    @classmethod
    def predict_batch(
        cls, features_list: List[List[float]], model_name: str = "default"
    ) -> List[Tuple[Any, Optional[float]]]:
        """
        Make batch predictions

        Args:
            features_list: List of feature vectors
            model_name: Name of the model to use

        Returns:
            List of (prediction, confidence) tuples

        Raises:
            ValueError: If model not found or input validation fails
        """
        # Input validation
        if not features_list:
            raise ValueError("Features list cannot be empty")
        if len(features_list) > 1000:  # Limit batch size to prevent DoS
            raise ValueError("Batch size too large (max 1000 predictions)")
        if not all(isinstance(f, list) for f in features_list):
            raise ValueError("All items must be feature lists")
        if not all(
            isinstance(x, (int, float)) and np.isfinite(x)
            for features in features_list
            for x in features
        ):
            raise ValueError("All features must be finite numeric values")

        # Validate model name to prevent injection
        # Allow alphanumeric, spaces, underscores, and hyphens (for team member names)
        if not isinstance(model_name, str) or not re.match(r"^[a-zA-Z0-9_\s-]+$", model_name):
            raise ValueError("Invalid model name")

        if model_name not in cls._models:
            raise ValueError(
                f"Model '{model_name}' not found. Available models: {list(cls._models.keys())}"
            )

        model = cls._models[model_name]
        features_array = np.array(features_list)

        # Make batch predictions
        predictions = model.predict(features_array)

        # Try to get prediction probabilities for confidence
        confidences = None
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(features_array)
                confidences = np.max(proba, axis=1)
            except Exception:
                pass

        # Convert numpy types to Python native types
        if isinstance(predictions, np.ndarray):
            predictions = predictions.tolist()
        else:
            predictions = [float(p) for p in predictions]

        if confidences is not None:
            confidences = confidences.tolist()
            return list(zip(predictions, confidences))
        else:
            return [(pred, None) for pred in predictions]

    @classmethod
    def list_models(cls) -> List[str]:
        """List all available model names"""
        return list(cls._models.keys())


class DummyModel:
    """
    Dummy model for demonstration purposes
    Returns a simple prediction based on feature sum
    """

    def predict(self, X):
        """Simple prediction: return sum of features"""
        return np.sum(X, axis=1)

    def predict_proba(self, X):
        """Simple probability: normalize feature sum"""
        pred = self.predict(X)
        # Convert to probabilities (simple normalization)
        prob = np.abs(pred) / (np.abs(pred) + 1)
        return np.column_stack([1 - prob, prob])
