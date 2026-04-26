import torch
import torchvision.models as models
from fastapi import FastAPI
from app.endpoints import router as vision_router
from contextlib import asynccontextmanager
import os

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading vision model...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    app.state.device = device
    
    # Load model
    model = models.mobilenet_v2()
    # Modify classifier to match our 10-class CIFAR10 finetuned model
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(num_ftrs, 10)
    
    model_path = 'model/vision_model.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Loaded finetuned weights.")
    else:
        print("Warning: Finetuned weights not found. Loading model with random classifier weights.")
    
    model = model.to(device)
    model.eval()
    app.state.model = model
    
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
    if hasattr(app.state, 'device'):
        del app.state.device

app = FastAPI(lifespan=lifespan, title="Vision Classification API")
app.include_router(vision_router)

@app.get("/")
def read_root():
    return {"status":"service functioning"}