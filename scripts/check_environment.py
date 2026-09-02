import sys


print("=" * 60)
print("FAKE NEWS DETECTION - ENVIRONMENT CHECK")
print("=" * 60)


print("\nPython")
print("-" * 60)
print(sys.version)


print("\nCore Libraries")
print("-" * 60)

import numpy
print("NumPy:", numpy.__version__)

import pandas
print("Pandas:", pandas.__version__)

import scipy
print("SciPy:", scipy.__version__)

import sklearn
print("Scikit-learn:", sklearn.__version__)

import joblib
print("Joblib:", joblib.__version__)


print("\nNLP")
print("-" * 60)

import nltk
print("NLTK:", nltk.__version__)

try:
    import spacy
    print("spaCy:", spacy.__version__)
except Exception as err:
    print(f"spaCy warning (optional dependencies): {err}")


print("\nBackend")
print("-" * 60)

import fastapi
print("FastAPI:", fastapi.__version__)

import uvicorn
print("Uvicorn:", uvicorn.__version__)


print("\nDatabase")
print("-" * 60)

import sqlite3
print("SQLite:", sqlite3.sqlite_version)

import sqlalchemy
print("SQLAlchemy:", sqlalchemy.__version__)


print("\nTesting")
print("-" * 60)

import pytest
print("Pytest:", pytest.__version__)


print("\nTransformer")
print("-" * 60)

try:
    import torch
    print("PyTorch:", torch.__version__)
    print("CUDA Available:", torch.cuda.is_available())

    import transformers
    print("Transformers:", transformers.__version__)
except Exception as err:
    print(f"PyTorch / Transformers load notice: {err}")
    print("Tip: Install Microsoft Visual C++ Redistributable (https://aka.ms/vs/17/release/vc_redist.x64.exe) if C++ DLLs are missing.")


print("\n" + "=" * 60)
print("ENVIRONMENT CHECK COMPLETED")
print("=" * 60)
