from fastapi import FastAPI
from app.endpoints import router as iris_router
from contextlib import asynccontextmanager
import joblib

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading model...")
    app.state.model = joblib.load('model/iris_model.pkl')
    yield
    print("Shutting down...")
    del app.state.model

app = FastAPI(lifespan=lifespan, title="Iris Prediction Service")
app.include_router(iris_router)
@app.get("/")
def read_root():
    return {"status":"service functioning"}