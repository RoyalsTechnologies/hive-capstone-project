"""
Health check endpoints
"""

from datetime import datetime

from fastapi import APIRouter

from app.services.model_service import ModelService

router = APIRouter()


@router.get("/health")
async def health_check():
    """Basic health check"""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@router.get("/health/ready")
async def readiness_check():
    """Readiness check - verifies models are loaded"""
    models_loaded = ModelService.are_models_loaded()
    return {
        "status": "ready" if models_loaded else "not ready",
        "models_loaded": models_loaded,
        "timestamp": datetime.utcnow().isoformat(),
    }
