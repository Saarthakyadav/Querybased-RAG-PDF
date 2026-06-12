import logging
import sys
from pathlib import Path
from typing import Any, Dict

from langchain_google_genai import ChatGoogleGenerativeAI

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluator import RAGEvaluator
from src.dataset_generator import DatasetGenerator

from config import EVAL_DATASET_PATH, EVAL_RESULTS_PATH

logger = logging.getLogger("docinsight.evaluation_service")


def run_evaluation() -> Dict[str, Any]:
    try:
        from app.services.rag_service import build_rag_runtime

        rag, _ = build_rag_runtime()
        gemini_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
        generator = DatasetGenerator(vector_store=rag.vector_store, llm=gemini_llm, rag_system=rag)
        dataset = generator.generate(num_questions=5, save_path=EVAL_DATASET_PATH)

        evaluator = RAGEvaluator(prefer_groq=False)
        results = evaluator.evaluate(dataset, save_path=EVAL_RESULTS_PATH, force_gemini=False)

        return {
            "status": "success",
            "dataset_size": len(dataset),
            "results": results,
        }
    except Exception as exc:
        logger.exception("Evaluation failed")
        raise RuntimeError(f"Evaluation failed: {exc}") from exc


def get_metrics() -> Dict[str, Any]:
    try:
        return RAGEvaluator.load(EVAL_RESULTS_PATH) or {"aggregate": {}, "per_sample": [], "num_samples": 0}
    except Exception as exc:
        logger.exception("Failed to load metrics")
        raise RuntimeError(f"Failed to load metrics: {exc}") from exc
