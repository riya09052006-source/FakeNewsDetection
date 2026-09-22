import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes import router
from backend.app.core.config import settings
from backend.app.core.logging_config import (
    configure_logging,
)
from backend.app.middleware.request_id import (
    RequestIDMiddleware,
)


# ============================================================
# LOGGING
# ============================================================

configure_logging()

logger = logging.getLogger(
    __name__
)


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description=settings.DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=[
        "GET",
        "POST",
    ],
    allow_headers=["*"],
)


# ============================================================
# REQUEST ID
# ============================================================

app.add_middleware(
    RequestIDMiddleware
)


# ============================================================
# ROUTES
# ============================================================

app.include_router(
    router
)


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
async def startup_event():

    logger.info(
        "%s starting...",
        settings.APP_NAME,
    )


# ============================================================
# SHUTDOWN
# ============================================================

@app.on_event("shutdown")
async def shutdown_event():

    logger.info(
        "%s shutting down...",
        settings.APP_NAME,
    )
