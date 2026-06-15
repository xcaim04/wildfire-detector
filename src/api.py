import numpy as np
from fastapi import FastAPI, HTTPException
from src.utils import PredictRequest, PredictResponse, create_and_train_model
from pydantic import BaseModel, Field


model = create_and_train_model()
app = FastAPI(title="Forest Fire API", version="1.0")

@app.get('/')
async def root():
    return {'Message': 'Welcome to Forest Fire API'}

