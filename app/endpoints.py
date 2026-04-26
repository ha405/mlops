from fastapi import APIRouter, Request, File, UploadFile, HTTPException
import os
import time
from PIL import Image
import io
import numpy as np
import onnxruntime as ort

router = APIRouter()

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
async def predict(request: Request, file: UploadFile = File(...)):
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

    # Return prediction result without MLflow logging
    return {
        "prediction": result, 
        "confidence": round(confidence, 4), 
        "latency_ms": round(latency, 2)
    }