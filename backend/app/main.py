import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes import router
from backend.app.core.config import settings
from backend.app.core.logging_config import (
    configure_logging,
)
from backend.app.database.init_db import (
    initialize_database,
)
from backend.app.middleware.request_id import (
    RequestIDMiddleware,
)
from backend.app.middleware.security_headers import (
    SecurityHeadersMiddleware,
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
    docs_url="/docs" if settings.ENABLE_DOCS else None,
    redoc_url="/redoc" if settings.ENABLE_DOCS else None,
)


# ============================================================
# SECURITY HEADERS
# ============================================================

app.add_middleware(
    SecurityHeadersMiddleware
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
        "DELETE",
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

    initialize_database()

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
