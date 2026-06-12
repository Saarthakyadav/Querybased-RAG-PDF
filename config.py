# ============================================================
# config.py
# ============================================================

# --- LLM ---
LLM_MODEL   = "meta-llama/llama-4-scout-17b-16e-instruct"
EMBED_MODEL = "all-MiniLM-L6-v2"

# --- Paths ---
VECTOR_STORE_DIR   = "data/vector_store"
PDF_DIR            = "data/pdf_files"
COLLECTION_NAME    = "pdf_documents"
EVAL_DATASET_PATH  = "data/eval_dataset.json"
EVAL_RESULTS_PATH  = "data/eval_results.json"

# --- Chunking ---
CHUNK_SIZE         = 1000
CHUNK_OVERLAP      = 150

# --- Embedding ---
EMBED_BATCH_SIZE   = 64
STORE_BATCH_SIZE   = 128

# --- Retrieval (dense) ---
TOP_K_DEFAULT      = 5
SCORE_THRESHOLD    = 0.35

# --- Hybrid retrieval ---
USE_HYBRID_RETRIEVAL = True   # set False to fall back to dense-only
USE_RERANKER         = True   # cross-encoder/ms-marco-MiniLM-L-6-v2
RRF_K                = 60     # RRF constant (higher = smoother fusion)
DENSE_WEIGHT         = 0.6    # weight in RRF score
BM25_WEIGHT          = 0.4

# --- Evaluation ---
EVAL_NUM_QUESTIONS       = 20    # synthetic Q&A pairs to generate
EVAL_QUESTIONS_PER_CHUNK = 2     # how many Qs to ask per source chunk
# NOTE: EVAL_JUDGE_MODEL removed — evaluator.py hardcodes Gemini 1.5 Flash.
# Update _build_gemini_llm() in evaluator.py directly if you need a different model.

# --- Upload limits ---
MAX_PDF_SIZE_MB = 50   # reject PDFs larger than this to prevent runaway ingestion