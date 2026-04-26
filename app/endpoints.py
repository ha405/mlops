from fastapi import APIRouter, Request, BackgroundTasks, File, UploadFile, HTTPException
import os
import mlflow
import time
import pandas as pd
from PIL import Image
import io
import torch
import torchvision.transforms as transforms

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

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@router.post("/predict")
async def predict(request: Request, bg: BackgroundTasks, file: UploadFile = File(...)):
    model = request.app.state.model
    class_names = request.app.state.class_names
    device = request.app.state.device
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")
        
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image file.")
        
    start = time.time()
    
    # Preprocess and predict
    input_tensor = transform(image)
    input_batch = input_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_batch)
        
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top_prob, top_catid = torch.topk(probabilities, 1)
    
    latency = (time.time() - start) * 1000
    
    result = class_names[top_catid[0].item()]
    confidence = top_prob[0].item()

    # Log to buffer
    log_entry = {
        "filename": file.filename,
        "content_type": file.content_type,
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