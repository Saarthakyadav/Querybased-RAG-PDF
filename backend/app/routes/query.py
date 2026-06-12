import logging

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from services.rag_service import query_documents

router = APIRouter(prefix="/query", tags=["query"])
logger = logging.getLogger("docinsight.routes.query")


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=10)
    score_threshold: float = Field(0.35, ge=0.0, le=1.0)
    use_hybrid: bool = True
    use_reranker: bool = True


@router.post("", status_code=status.HTTP_200_OK)
async def run_query(payload: QueryRequest):
    try:
        result = query_documents(
            question=payload.question,
            top_k=payload.top_k,
            score_threshold=payload.score_threshold,
            use_hybrid=payload.use_hybrid,
            use_reranker=payload.use_reranker,
        )
        return {"status": "ok", "data": result}
    except Exception as exc:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
