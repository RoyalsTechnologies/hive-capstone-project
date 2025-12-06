"""
Weather Inference endpoints
"""

import logging
import re
from typing import Any, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from app.services.model_service import ModelService

logger = logging.getLogger(__name__)

router = APIRouter()


class PredictionRequest(BaseModel):
    """Request model for predictions"""

    model_config = {"protected_namespaces": ()}

    features: List[float] = Field(..., min_length=1, max_length=1000, description="Feature values")
    model_name: str = Field(default="default", description="Model name")

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """Validate model name contains only safe characters"""
        # Allow alphanumeric, spaces, underscores, and hyphens for team member names
        if not re.match(r"^[a-zA-Z0-9_\s-]+$", v):
            raise ValueError(
                "Model name must contain only alphanumeric characters, "
                "spaces, underscores, and hyphens"
            )
        return v.strip()  # Trim whitespace


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""

    model_config = {"protected_namespaces": ()}

    features: List[List[float]] = Field(
        ..., min_length=1, max_length=1000, description="List of feature vectors"
    )
    model_name: str = Field(default="default", description="Model name")

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """Validate model name contains only safe characters"""
        # Allow alphanumeric, spaces, underscores, and hyphens for team member names
        if not re.match(r"^[a-zA-Z0-9_\s-]+$", v):
            raise ValueError(
                "Model name must contain only alphanumeric characters, "
                "spaces, underscores, and hyphens"
            )
        return v.strip()  # Trim whitespace


class PredictionResponse(BaseModel):
    """Response model for predictions"""

    prediction: Any
    model_name: str
    confidence: float = None


@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Single prediction endpoint

    Args:
        request: PredictionRequest with features and optional model_name

    Returns:
        PredictionResponse with prediction result
    """
    try:
        if not ModelService.are_models_loaded():
            raise HTTPException(
                status_code=503,
                detail="Models are not loaded. Please wait for initialization.",
            )

        prediction, confidence = ModelService.predict(
            features=request.features, model_name=request.model_name
        )

        return PredictionResponse(
            prediction=prediction, model_name=request.model_name, confidence=confidence
        )
    except ValueError as e:
        # Input validation errors - safe to expose
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Don't expose internal error details in production
        import logging

        logger = logging.getLogger(__name__)
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="An internal error occurred. Please try again later."
        )


@router.post("/predict/batch", response_model=List[PredictionResponse])
async def predict_batch(request: BatchPredictionRequest):
    """
    Batch prediction endpoint

    Args:
        request: BatchPredictionRequest with list of feature vectors

    Returns:
        List of PredictionResponse objects
    """
    try:
        if not ModelService.are_models_loaded():
            raise HTTPException(
                status_code=503,
                detail="Models are not loaded. Please wait for initialization.",
            )

        predictions = ModelService.predict_batch(
            features_list=request.features, model_name=request.model_name
        )

        return [
            PredictionResponse(prediction=pred, model_name=request.model_name, confidence=conf)
            for pred, conf in predictions
        ]
    except ValueError as e:
        # Input validation errors - safe to expose
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Don't expose internal error details in production
        logger.error(f"Batch prediction failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="An internal error occurred. Please try again later."
        )


@router.get("/models")
async def list_models():
    """List available models"""
    return {
        "available_models": ModelService.list_models(),
        "models_loaded": ModelService.are_models_loaded(),
    }
