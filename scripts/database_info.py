from backend.app.database.database import (
    SessionLocal,
)

from backend.app.database.models import (
    PredictionHistory,
)


def main():

    db = SessionLocal()

    try:

        count = (
            db.query(
                PredictionHistory
            ).count()
        )

        print(
            "Stored predictions:",
            count,
        )

        latest = (
            db.query(
                PredictionHistory
            )
            .order_by(
                PredictionHistory.created_at.desc()
            )
            .first()
        )

        if latest:

            print(
                "Latest prediction:",
                latest.prediction,
            )

            print(
                "Latest score:",
                latest.decision_score,
            )

        else:

            print(
                "No predictions stored."
            )

    finally:

        db.close()


if __name__ == "__main__":
    main()
