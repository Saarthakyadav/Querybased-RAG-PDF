import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.routes.upload import router as upload_router
from app.routes.query import router as query_router
from app.routes.evaluate import router as evaluate_router
from app.routes.metrics import router as metrics_router
from app.routes.health import router as health_router

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("docinsight_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting DocInsight AI backend")
    yield
    logger.info("Shutting down DocInsight AI backend")


app = FastAPI(
    title="DocInsight AI API",
    description="FastAPI wrapper around the existing RAG pipeline",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:5173").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(upload_router, prefix="/api")
app.include_router(query_router, prefix="/api")
app.include_router(evaluate_router, prefix="/api")
app.include_router(metrics_router, prefix="/api")


@app.get("/")
def root():
    return {"message": "DocInsight AI backend is running"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
