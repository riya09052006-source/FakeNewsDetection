from backend.app.database.database import Base
from backend.app.database.database import engine

from backend.app.database import models


def initialize_database():
    Base.metadata.create_all(
        bind=engine
    )


if __name__ == "__main__":
    initialize_database()

    print(
        "Database initialized successfully."
    )
