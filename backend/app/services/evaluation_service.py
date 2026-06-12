import logging
from typing import Any, Dict, List

from src.evaluator import RAGEvaluator
from src.dataset_generator import DatasetGenerator

from config import EVAL_DATASET_PATH, EVAL_RESULTS_PATH

logger = logging.getLogger("docinsight.evaluation_service")


def run_evaluation() -> Dict[str, Any]:
    try:
        from services.rag_service import build_rag_runtime

        rag, _ = build_rag_runtime()
        generator = DatasetGenerator(vector_store=rag.vector_store, llm=rag.llm, rag_system=rag)
        dataset = generator.generate(num_questions=5, save_path=EVAL_DATASET_PATH)

        evaluator = RAGEvaluator(prefer_groq=True)
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
