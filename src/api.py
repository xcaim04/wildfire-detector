from fastapi import FastAPI
from src.routers.health import router as health_router
from src.routers.wildfire import router as wildfire_router

app = FastAPI(
    title="Wildfire Risk AI Prediction API",
    description="FastAPI gateway integrated with Pytorch.",
    version="1.0.0",
)

app.include_router(health_router)
app.include_router(wildfire_router)
