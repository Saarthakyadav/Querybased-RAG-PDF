import logging
from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from services.rag_service import clear_documents, ingest_pdfs

router = APIRouter(prefix="/upload", tags=["upload"])
logger = logging.getLogger("docinsight.routes.upload")


@router.post("", status_code=status.HTTP_200_OK)
async def upload_pdfs(files: List[UploadFile] = File(...)):
    try:
        result = ingest_pdfs(files)
        return {"status": "ok", **result}
    except Exception as exc:
        logger.exception("Upload failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.delete("", status_code=status.HTTP_200_OK)
async def reset_documents():
    try:
        return clear_documents()
    except Exception as exc:
        logger.exception("Reset failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
