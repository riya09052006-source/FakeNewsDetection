import json
import logging
from pathlib import Path

from fastapi import (
    APIRouter,
    HTTPException,
    Request,
)

from backend.app.schemas.prediction import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)

from backend.app.services.prediction_service import (
    PredictionService,
)


logger = logging.getLogger(
    __name__
)

router = APIRouter()

prediction_service = PredictionService()


# ============================================================
# ROOT
# ============================================================

@router.get("/")
def root():

    return {
        "application": "Fake News Detection API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
    }


# ============================================================
# HEALTH
# ============================================================

@router.get(
    "/health",
    response_model=HealthResponse,
)
def health():

    ready = prediction_service.is_ready()

    return {
        "status": "healthy" if ready else "degraded",
        "model_ready": ready,
    }


# ============================================================
# MODEL INFORMATION
# ============================================================

@router.get(
    "/model-info",
    response_model=ModelInfoResponse,
)
def model_info():

    base_dir = Path(__file__).resolve().parents[3]

    metadata_path = (
        base_dir /
        "models" /
        "optimized" /
        "model_metadata.json"
    )

    if not metadata_path.exists():

        raise HTTPException(
            status_code=404,
            detail="Model metadata not found.",
        )

    try:

        with open(
            metadata_path,
            "r",
            encoding="utf-8",
        ) as file:

            metadata = json.load(file)

    except Exception as error:

        logger.exception(
            "Failed to load model metadata."
        )

        raise HTTPException(
            status_code=500,
            detail="Unable to load model metadata.",
        ) from error

    return {
        "model": metadata.get(
            "model",
            "Optimized Linear SVM",
        ),
        "feature_type": metadata.get(
            "feature_type",
            "Word TF-IDF + Character TF-IDF",
        ),
        "prediction_labels": metadata.get(
            "prediction_labels",
            {
                "0": "FAKE",
                "1": "REAL",
            },
        ),
        "warning": metadata.get(
            "warning",
            (
                "Predictions are model classifications "
                "and are not independent factual verification."
            ),
        ),
    }


# ============================================================
# SINGLE PREDICTION
# ============================================================

@router.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    payload: PredictionRequest,
    request: Request,
):

    request_id = getattr(
        request.state,
        "request_id",
        None,
    )

    try:

        result = prediction_service.predict(
            payload.text
        )

        result["request_id"] = request_id

        logger.info(
            "Prediction completed | request_id=%s | prediction=%s",
            request_id,
            result["prediction"],
        )

        return result

    except (
        TypeError,
        ValueError,
    ) as error:

        logger.warning(
            "Invalid prediction request | request_id=%s | error=%s",
            request_id,
            error,
        )

        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except Exception as error:

        logger.exception(
            "Prediction failed | request_id=%s",
            request_id,
        )

        raise HTTPException(
            status_code=500,
            detail="Prediction service failed.",
        ) from error


# ============================================================
# BATCH PREDICTION
# ============================================================

@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
)
def predict_batch(
    payload: BatchPredictionRequest,
    request: Request,
):

    request_id = getattr(
        request.state,
        "request_id",
        None,
    )

    try:

        results = prediction_service.predict_batch(
            payload.texts
        )

        logger.info(
            "Batch prediction completed | request_id=%s | count=%s",
            request_id,
            len(results),
        )

        return {
            "results": results,
            "count": len(results),
            "request_id": request_id,
        }

    except Exception as error:

        logger.exception(
            "Batch prediction failed | request_id=%s",
            request_id,
        )

        raise HTTPException(
            status_code=500,
            detail="Batch prediction failed.",
        ) from error
