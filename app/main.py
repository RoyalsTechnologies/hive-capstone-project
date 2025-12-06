"""
Hive Weather Inference API
Main application entry point
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from app.config import settings
from app.routes import health, inference

app = FastAPI(
    title="Hive Weather Inference API",
    description="Hive Weather Inference Service with FastAPI",
    version="1.0.0",
)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses"""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        # Only add HSTS in production with HTTPS
        if not settings.DEBUG:
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response


# Security headers middleware (add before CORS)
app.add_middleware(SecurityHeadersMiddleware)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(inference.router, prefix="/api/v1", tags=["Inference"])

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/home")
async def home():
    """Serve the Hive Weather Inference UI"""
    return FileResponse("app/static/index.html")


@app.on_event("startup")
async def startup_event():
    """Initialize weather inference models on startup"""
    from app.services.model_service import ModelService

    ModelService.load_models()


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    from app.services.model_service import ModelService

    ModelService.unload_models()


@app.get("/")
async def root():
    """Root endpoint - redirects to home"""
    from fastapi.responses import RedirectResponse

    return RedirectResponse(url="/home")
