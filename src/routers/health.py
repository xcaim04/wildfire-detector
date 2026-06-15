from fastapi import APIRouter
from ai.model import model

router = APIRouter()

@router.get("/health")
async def health():
    return {"status": "ok", "model_ready": model is not None}
