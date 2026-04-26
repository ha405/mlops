from fastapi import APIRouter, Request, BackgroundTasks, File, UploadFile, HTTPException
import os
import mlflow
import time
import pandas as pd
from PIL import Image
import io
import numpy as np
import onnxruntime as ort

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

# Setup MLflow optionally
def init_mlflow():
    try:
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment("vision_classification_service")
        print(f"MLflow initialized at {MLFLOW_URI}")
        return True
    except Exception as e:
        print(f"MLflow initialization failed: {e}")
        return False

MLFLOW_CONFIGURED = init_mlflow()

RESULTS_BUFFER = []
BUFFER_THRESHOLD = 10
router = APIRouter()

def flush_to_mlflow(data):
    df = pd.DataFrame(data)
    with mlflow.start_run(run_name="buffer_flush"):
        mlflow.log_table(data=df, artifact_file="predictions.json")

def preprocess_image(image: Image.Image):
    """Preprocess image for ONNX model without using torch."""
    image = image.resize((224, 224))
    img_data = np.array(image).astype('float32')
    
    # Normalize (ImageNet stats)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_data = (img_data / 255.0 - mean) / std
    
    # Transpose to [C, H, W]
    img_data = img_data.transpose(2, 0, 1)
    
    # Add batch dimension [1, C, H, W]
    img_data = np.expand_dims(img_data, axis=0)
    return img_data

@router.post("/predict")
async def predict(request: Request, bg: BackgroundTasks, file: UploadFile = File(...)):
    session = request.app.state.model
    class_names = request.app.state.class_names
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")
        
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image file.")
        
    start = time.time()
    
    # Preprocess
    input_data = preprocess_image(image)
    
    # Predict using ONNX
    inputs = {session.get_inputs()[0].name: input_data}
    outputs = session.run(None, inputs)
    output = outputs[0]
    
    # Post-process (Softmax + Topk)
    exp_output = np.exp(output[0] - np.max(output[0]))
    probabilities = exp_output / exp_output.sum()
    top_catid = np.argmax(probabilities)
    confidence = float(probabilities[top_catid])
    
    latency = (time.time() - start) * 1000
    result = class_names[top_catid]

    # Log to buffer
    log_entry = {
        "filename": file.filename,
        "prediction": result,
        "confidence": confidence,
        "latency_ms": latency
    }
    RESULTS_BUFFER.append(log_entry)

    if len(RESULTS_BUFFER) >= BUFFER_THRESHOLD:
        to_flush = RESULTS_BUFFER.copy()
        RESULTS_BUFFER.clear()
        bg.add_task(flush_to_mlflow, to_flush)
        
    return {"prediction": result, "confidence": confidence, "latency_ms": latency}