import logging

from fastapi import APIRouter, HTTPException, status

from services.evaluation_service import get_metrics, run_evaluation

router = APIRouter(prefix="/evaluate", tags=["evaluate"])
logger = logging.getLogger("docinsight.routes.evaluate")


@router.post("", status_code=status.HTTP_200_OK)
async def evaluate_documents():
    try:
        result = run_evaluation()
        return {"status": "ok", "data": result}
    except Exception as exc:
        logger.exception("Evaluation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/metrics", status_code=status.HTTP_200_OK)
async def metrics():
    try:
        return {"status": "ok", "data": get_metrics()}
    except Exception as exc:
        logger.exception("Metrics retrieval failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
