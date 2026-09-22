from datetime import datetime

from sqlalchemy import Column
from sqlalchemy import DateTime
from sqlalchemy import Float
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy import Text

from backend.app.database.database import Base


class PredictionHistory(Base):

    __tablename__ = "prediction_history"

    id = Column(
        Integer,
        primary_key=True,
        index=True,
    )

    text = Column(
        Text,
        nullable=False,
    )

    prediction = Column(
        String(10),
        nullable=False,
    )

    label_id = Column(
        Integer,
        nullable=False,
    )

    decision_score = Column(
        Float,
        nullable=False,
    )

    model = Column(
        String(100),
        nullable=False,
    )

    feature_type = Column(
        String(200),
        nullable=False,
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )
