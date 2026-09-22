from sqlalchemy.orm import Session

from backend.app.database.models import (
    PredictionHistory,
)


def save_prediction(
    db: Session,
    result: dict,
    text: str,
):

    record = PredictionHistory(
        text=text,
        prediction=result["prediction"],
        label_id=result["label_id"],
        decision_score=result["decision_score"],
        model=result["model"],
        feature_type=result["feature_type"],
    )

    db.add(record)

    db.commit()

    db.refresh(record)

    return record


def get_predictions(
    db: Session,
    limit: int = 50,
):

    return (
        db.query(PredictionHistory)
        .order_by(
            PredictionHistory.created_at.desc()
        )
        .limit(limit)
        .all()
    )


def get_prediction(
    db: Session,
    prediction_id: int,
):

    return (
        db.query(PredictionHistory)
        .filter(
            PredictionHistory.id
            == prediction_id
        )
        .first()
    )


def delete_all_predictions(
    db: Session,
):

    deleted_count = (
        db.query(PredictionHistory).delete()
    )

    db.commit()

    return deleted_count


def get_analytics(
    db: Session,
):

    total = (
        db.query(PredictionHistory).count()
    )

    fake_count = (
        db.query(PredictionHistory)
        .filter(
            PredictionHistory.label_id == 0
        )
        .count()
    )

    real_count = (
        db.query(PredictionHistory)
        .filter(
            PredictionHistory.label_id == 1
        )
        .count()
    )

    if total > 0:

        fake_percentage = (
            fake_count / total
        ) * 100

        real_percentage = (
            real_count / total
        ) * 100

    else:

        fake_percentage = 0.0
        real_percentage = 0.0

    return {
        "total": total,
        "fake_count": fake_count,
        "real_count": real_count,
        "fake_percentage": fake_percentage,
        "real_percentage": real_percentage,
    }
