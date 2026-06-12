import logging
import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from routes.upload import router as upload_router
from routes.query import router as query_router
from routes.evaluate import router as evaluate_router
from routes.health import router as health_router

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


@app.get("/")
def root():
    return {"message": "DocInsight AI backend is running"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
