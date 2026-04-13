from fastapi import APIRouter, Request, BackgroundTasks
from pydantic import BaseModel
import os
import pkg_resources
import mlflow
import time
import pandas as pd

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("iris_prediction_service")

RESULTS_BUFFER = []
BUFFER_THRESHOLD = 10
router = APIRouter()

class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

def flush_to_mlflow(data):
    df = pd.DataFrame(data)
    with mlflow.start_run(run_name="buffer_flush"):
        mlflow.log_table(data=df, artifact_file="predictions.json")

@router.post("/predict")
async def predict(request: Request, data: IrisFeatures, bg: BackgroundTasks):
    model = request.app.state.model
    features = [[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]]

    start = time.time()
    prediction = model.predict(features)
    latency = (time.time() - start) * 1000

    labels = ['setosa', 'versicolor', 'virginica']
    result = labels[prediction[0]]

    log_entry = data.model_dump()
    log_entry.update({"prediction": result, "latency_ms": latency})
    RESULTS_BUFFER.append(log_entry)

    if len(RESULTS_BUFFER) >= BUFFER_THRESHOLD:
        to_flush = RESULTS_BUFFER.copy()
        RESULTS_BUFFER.clear()
        bg.add_task(flush_to_mlflow, to_flush)
    return {"prediction": result, "latency_ms": latency}