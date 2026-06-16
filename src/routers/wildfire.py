import os
import torch
import joblib
import pandas as pd
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from src.ai.model.model import WildfireNet
from src.ai.data.data import load_and_process_data
from src.ai.model.train import train_model

router = APIRouter(prefix="/wildfire", tags=["Wildfire Prediction System"])

# Paths setup based on the project layout
AI_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ai"))
SCALER_PATH = os.path.join(AI_DIR, 'data', 'scaler.pkl')
MODEL_PATH = os.path.join(AI_DIR, 'model', 'wildfire_model.pth')

scaler = None
model = None
is_processing_data = False
is_training_model = False

def init_prediction_model(force_reload=False):
    global scaler, model
    if not force_reload and scaler is not None and model is not None:
        return

    if not os.path.exists(SCALER_PATH) or not os.path.exists(MODEL_PATH):
        raise RuntimeError("Model or Scaler missing. Run data processing and training first.")
        
    scaler = joblib.load(SCALER_PATH)
    model = WildfireNet()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()

class WildfireFeaturesInput(BaseModel):
    X: int = Field(..., ge=1, le=9)
    Y: int = Field(..., ge=1, le=9)
    month: str
    day: str
    FFMC: float
    DMC: float
    DC: float
    ISI: float
    temp: float
    RH: float
    wind: float
    rain: float

def run_background_train():
    global is_training_model
    try:
        train_model()
        init_prediction_model(force_reload=True)
    finally:
        is_training_model = False

@router.post("/process-data")
async def process_dataset():
    global is_processing_data
    if is_processing_data:
        raise HTTPException(status_code=409, detail="Data processing task is already running.")
    try:
        is_processing_data = True
        X_train, X_test, y_train, y_test = load_and_process_data()
        return {
            "status": "success",
            "message": "Dataset processed successfully.",
            "data_summary": {"train_samples": X_train.shape[0], "test_samples": X_test.shape[0]}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        is_processing_data = False

@router.post("/train")
async def trigger_training(background_tasks: BackgroundTasks):
    global is_training_model
    if is_training_model:
        return {"status": "processing", "message": "Model training is already in progress."}

    is_training_model = True
    background_tasks.add_task(run_background_train)
    return {"status": "started", "message": "Model training initiated in the background."}

@router.get("/status")
async def get_system_status():
    return {
        "data_processing_active": is_processing_data,
        "training_active": is_training_model,
        "model_loaded": model is not None,
        "weights_exist": os.path.exists(MODEL_PATH)
    }

@router.post("/predict")
async def predict_wildfire(data: WildfireFeaturesInput):
    try:
        init_prediction_model()
        months_map = {'jan':1, 'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6, 'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12}
        days_map = {'mon':1, 'tue':2, 'wed':3, 'thu':4, 'fri':5, 'sat':6, 'sun':7}
        
        month_lower, day_lower = data.month.lower()[:3], data.day.lower()[:3]
        if month_lower not in months_map or day_lower not in days_map:
            raise HTTPException(status_code=400, detail="Invalid date parameters.")
            
        features_dict = {
            'X': data.X, 'Y': data.Y, 'month': months_map[month_lower], 'day': days_map[day_lower],
            'FFMC': data.FFMC, 'DMC': data.DMC, 'DC': data.DC, 'ISI': data.ISI,
            'temp': data.temp, 'RH': data.RH, 'wind': data.wind, 'rain': data.rain
        }
        
        df_features = pd.DataFrame([features_dict])
        scaled_features = scaler.transform(df_features.values)
        input_tensor = torch.tensor(scaled_features, dtype=torch.float32)
        
        with torch.no_grad():
            output_probability = model(input_tensor).item()
            
        return {
            "status": "success",
            "fire_prediction": bool(output_probability >= 0.5),
            "fire_probability": round(output_probability, 4)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
