import numpy as np
from fastapi import FastAPI, HTTPException
from src.utils import PredictRequest, PredictResponse, create_and_train_model
from pydantic import BaseModel, Field


model = create_and_train_model()
app = FastAPI(title="Polynomial Neural Network API", version="1.0")

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    try:
        X = np.array(request.x).reshape(-1, 1)
        y_pred = model.predict(X)
        predictions = y_pred.flatten().tolist()
        return PredictResponse(predictions=predictions)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
