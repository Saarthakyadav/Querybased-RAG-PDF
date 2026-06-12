from fastapi import APIRouter, HTTPException, status

from app.services.evaluation_service import get_metrics

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get("", status_code=status.HTTP_200_OK)
async def metrics():
    try:
        return {"status": "ok", "data": get_metrics()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
