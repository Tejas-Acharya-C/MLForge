"""MLForge FastAPI Application Factory."""

import logging
from typing import Dict, Set

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from mlforge.api.routes import router as api_router
from mlforge.core.exceptions import MLForgeError

logger = logging.getLogger("mlforge.api.app")

ALLOWED_LOCAL_ORIGINS: Set[str] = {
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
}

ALLOWED_HOSTS: Set[str] = {"127.0.0.1", "localhost", "::1", "testserver"}


def create_app() -> FastAPI:
    """Create and configure the MLForge FastAPI application instance."""
    app = FastAPI(
        title="MLForge Laboratory API",
        version="0.1.0",
        description="Local-first AI Model Evaluation & Reliability Lab for tabular classification.",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # 1. Security middleware: DNS-rebinding guard & browser state-changing Origin validation
    @app.middleware("http")
    async def security_host_and_origin_guard(request: Request, call_next):
        # Validate Host header against DNS rebinding
        host_header = request.headers.get("host", "").split(":")[0].lower()
        if host_header and host_header not in ALLOWED_HOSTS:
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={"error": "SecurityRebindingError", "detail": f"Untrusted host header '{host_header}' rejected."},
            )

        # Validate Origin header on state-changing requests from browsers
        if request.method in ("POST", "PUT", "DELETE", "PATCH"):
            origin = request.headers.get("origin")
            if origin and origin not in ALLOWED_LOCAL_ORIGINS:
                return JSONResponse(
                    status_code=status.HTTP_403_FORBIDDEN,
                    content={"error": "SecurityOriginError", "detail": f"Cross-origin request from '{origin}' forbidden."},
                )

        return await call_next(request)

    # 2. Configure CORS for local frontend development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(ALLOWED_LOCAL_ORIGINS),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 3. Global custom exception handler for MLForge core errors
    @app.exception_handler(MLForgeError)
    async def mlforge_error_handler(request: Request, exc: MLForgeError) -> JSONResponse:
        logger.warning(f"Handled MLForge domain error on {request.method} {request.url.path}: {exc}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": exc.__class__.__name__,
                "detail": str(exc),
            },
        )

    # 4. Include API v1 router
    app.include_router(api_router)

    # 5. Root health / identity endpoint
    @app.get("/", tags=["Health"])
    def root_health() -> Dict[str, str]:
        return {
            "application": "MLForge",
            "environment": "local-development",
            "status": "ready",
            "version": "0.1.0",
            "docs": "/docs",
        }

    # 6. Standard healthz endpoint for UI connectivity verification
    @app.get("/healthz", tags=["Health"])
    def health_check() -> Dict[str, str]:
        return {
            "status": "ok",
            "service": "mlforge",
            "version": "0.1.0",
        }

    return app


app = create_app()
