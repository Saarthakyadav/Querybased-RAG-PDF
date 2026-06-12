import logging
from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.services.rag_service import clear_documents, ingest_pdfs
from config import MAX_PDF_SIZE_MB

router = APIRouter(prefix="/upload", tags=["upload"])
logger = logging.getLogger("docinsight.routes.upload")

_MAX_BYTES = MAX_PDF_SIZE_MB * 1024 * 1024


@router.post("", status_code=status.HTTP_200_OK)
async def upload_pdfs(files: List[UploadFile] = File(...)):
    # FIX: enforce file size limit to prevent runaway ingestion on huge PDFs.
    for f in files:
        contents = await f.read()
        if len(contents) > _MAX_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"{f.filename} exceeds the {MAX_PDF_SIZE_MB} MB limit.",
            )
        await f.seek(0)  # reset so ingest_pdfs can re-read

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