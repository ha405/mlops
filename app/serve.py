import onnxruntime as ort
from fastapi import FastAPI
from app.endpoints import router as vision_router
from contextlib import asynccontextmanager
import os

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading ONNX vision model...")
    
    model_path = 'model/vision_model.onnx'
    if not os.path.exists(model_path):
        # Fallback to .pth if .onnx isn't ready, but warn
        print(f"Warning: {model_path} not found. Ensure training script has run.")
        # We need a model to start, so let's expect the user to have run training.
        # In a real app, we might download a default model here.
    
    try:
        # Load ONNX session
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        app.state.model = session
        print("ONNX model loaded successfully.")
    except Exception as e:
        print(f"Error loading ONNX model: {e}")
        app.state.model = None

    # Load classes
    classes_path = 'model/cifar10_classes.txt'
    if os.path.exists(classes_path):
        with open(classes_path, 'r') as f:
            app.state.class_names = [line.strip() for line in f.readlines()]
    else:
        app.state.class_names = [f"Class {i}" for i in range(10)]
        
    yield
    print("Shutting down...")
    if hasattr(app.state, 'model'):
        del app.state.model
    if hasattr(app.state, 'class_names'):
        del app.state.class_names

app = FastAPI(lifespan=lifespan, title="Optimized Vision API (ONNX)")
app.include_router(vision_router)

@app.get("/")
def read_root():
    return {"status":"service functioning"}