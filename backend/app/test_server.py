from fastapi import FastAPI


app = FastAPI()


@app.get("/")
def home():

    return {
        "message": "Fake News Detection Backend Working!"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy"
    }
