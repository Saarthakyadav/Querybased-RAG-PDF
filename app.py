# ============================================================
# app.py  (updated — adds Evaluation tab)
# ============================================================

import os
import time
import shutil
import json
from typing import List

import streamlit as st



from dotenv import load_dotenv
from langchain_groq import ChatGroq

from src.data_loader       import PDFProcessor
from src.embedding         import EmbeddingManager
from src.vectorstore       import VectorStore
from src.rag_system        import RAGSystem
from src.image_processor   import ImageProcessor
from src.dataset_generator import DatasetGenerator
from src.evaluator         import RAGEvaluator

from config import (
    LLM_MODEL,
    EMBED_BATCH_SIZE,
    STORE_BATCH_SIZE,
    PDF_DIR,
    TOP_K_DEFAULT,
    SCORE_THRESHOLD,
    USE_HYBRID_RETRIEVAL,
    USE_RERANKER,
    EVAL_DATASET_PATH,
    EVAL_RESULTS_PATH,
    EVAL_NUM_QUESTIONS,
)

load_dotenv()

# ---------------------------------------------------------------
# Page config
# ---------------------------------------------------------------
st.set_page_config(page_title="PDF RAG Assistant", layout="wide")
st.title("📚 Research Paper Q&A")

# ---------------------------------------------------------------
# Backend init (cached)
# ---------------------------------------------------------------
@st.cache_resource
def init_rag():
    llm        = ChatGroq(model=LLM_MODEL)
    embeddings = EmbeddingManager(batch_size=EMBED_BATCH_SIZE)
    store      = VectorStore(batch_size=STORE_BATCH_SIZE)
    image_proc = ImageProcessor()
    rag = RAGSystem(
        store, embeddings, llm,
        use_hybrid=USE_HYBRID_RETRIEVAL,
        use_reranker=USE_RERANKER,
    )
    return rag, image_proc

rag, image_processor = init_rag()

# ---------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------
if "messages"            not in st.session_state:
    st.session_state.messages = []
if "eval_dataset"        not in st.session_state:
    st.session_state.eval_dataset = None
if "eval_results"        not in st.session_state:
    st.session_state.eval_results = None
if "baseline_results"    not in st.session_state:
    st.session_state.baseline_results = None

# ---------------------------------------------------------------
# Sidebar — document ingestion (shared across tabs)
# ---------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")

    top_k     = st.slider("Top K (context chunks)", 1, 10, TOP_K_DEFAULT)
    threshold = st.slider("Similarity threshold",   0.0, 1.0, SCORE_THRESHOLD)

    st.markdown("---")

    use_hybrid  = st.toggle("🔀 Hybrid retrieval (BM25 + dense)", value=USE_HYBRID_RETRIEVAL)
    use_reranker = st.toggle("🏅 Cross-encoder reranker",          value=USE_RERANKER)

    # Update RAG system live if toggles change
    rag._hybrid_retriever._reranker = None  # will reload on next call if needed
    rag.use_hybrid = use_hybrid
    rag._hybrid_retriever.use_reranker = use_reranker

    st.markdown("---")

    uploaded_files = st.file_uploader(
        "Upload PDF documents", type="pdf", accept_multiple_files=True
    )

    if st.button("⚡ Process & index documents"):
        if uploaded_files:
            os.makedirs(PDF_DIR, exist_ok=True)
            for uploaded_file in uploaded_files:
                dest = os.path.join(PDF_DIR, uploaded_file.name)
                with open(dest, "wb") as f:
                    f.write(uploaded_file.getbuffer())

            with st.spinner("Ingesting…"):
                start_total = time.time()

                processor = PDFProcessor(
                    llm=rag.llm,
                    vision_func=image_processor.get_image_description,
                    process_images=True,
                )

                t1       = time.time()
                raw_docs = processor.process_pdfs(PDF_DIR)
                st.info(f"📄 Extracted {len(raw_docs)} pages in {time.time()-t1:.2f}s")

                t2     = time.time()
                chunks = processor.split_documents(raw_docs)
                st.info(f"✂️ Created {len(chunks)} chunks in {time.time()-t2:.2f}s")

                t3       = time.time()
                contents = [doc.page_content for doc in chunks]
                embeddings_arr, valid_indices = rag.embedding_manager.generate_embeddings(contents)
                chunks = [chunks[i] for i in valid_indices]
                st.info(f"🧠 Embeddings generated in {time.time()-t3:.2f}s")

                assert len(chunks) == len(embeddings_arr)

                t4 = time.time()
                rag.vector_store.add_documents(chunks, embeddings_arr)
                st.info(f"💾 Stored in {time.time()-t4:.2f}s")
                st.success(f"✅ Done in {time.time()-start_total:.2f}s")
        else:
            st.warning("Please upload at least one PDF first.")

    st.markdown("---")

    if st.button("🗑️ Clear all documents"):
        shutil.rmtree(PDF_DIR, ignore_errors=True)
        os.makedirs(PDF_DIR, exist_ok=True)
        rag.vector_store.reset()
        st.session_state.messages      = []
        st.session_state.eval_dataset  = None
        st.session_state.eval_results  = None
        st.session_state.baseline_results = None
        st.success("Cleared. Re-upload your documents.")

# ---------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------
tab_chat, tab_eval = st.tabs(["💬 Chat", "📊 Evaluation"])

# ===============================================================
# TAB 1 — Chat (original behaviour, unchanged)
# ===============================================================
with tab_chat:
    # Render history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📚 Source chunks"):
                    for idx, (source, chunk) in enumerate(
                        zip(msg["sources"], msg["chunks"])
                    ):
                        st.markdown(
                            f"**Source {idx+1}** · Page {source.get('page')} "
                            f"· `{source.get('source_file')}`"
                        )
                        st.caption(chunk["content"][:400])
                        if idx < len(msg["sources"]) - 1:
                            st.divider()

    prompt = st.chat_input("Ask a question about your documents…")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                result = rag.ask(prompt, top_k=top_k, score_threshold=threshold)
                answer = result["answer"]

            st.markdown(answer)

            if result["sources"]:
                m1, m2, m3, m4 = st.columns(4)
                first = result["sources"][0]
                m1.metric("Author",          first.get("author",  "N/A"))
                m2.metric("Date",            first.get("date",    "N/A"))
                m3.metric("Relevance score", result["metrics"].get("relevance_score", "0%"))
                m4.metric("Time taken",      result["metrics"].get("fetch_time",      "0s"))

            if result["sources"]:
                with st.expander("📚 Source chunks"):
                    for idx, (source, chunk) in enumerate(
                        zip(result["sources"], result["chunks"])
                    ):
                        st.markdown(
                            f"**Source {idx+1}** · Page {source.get('page')} "
                            f"· `{source.get('source_file')}`"
                        )
                        st.caption(chunk["content"][:400])
                        if idx < len(result["sources"]) - 1:
                            st.divider()

        st.session_state.messages.append({
            "role":    "assistant",
            "content": answer,
            "sources": result["sources"],
            "chunks":  result["chunks"],
        })

# ===============================================================
# TAB 2 — Evaluation
# ===============================================================
with tab_eval:
    st.header("📊 RAGAS Evaluation")
    st.caption(
        "Measures retrieval and generation quality using 4 RAGAS metrics: "
        "Faithfulness, Answer Relevancy, Context Precision, Context Recall."
    )

    doc_count = rag.vector_store.collection.count()
    if doc_count == 0:
        st.warning("⚠️ No documents indexed yet. Upload and process PDFs first.")
        st.stop()

    st.info(f"✅ {doc_count} chunks indexed — ready for evaluation.")

    # ---- Step 1: Generate dataset ----
    st.subheader("Step 1 — Generate Evaluation Dataset")

    col1, col2 = st.columns([2, 1])
    with col1:
        n_questions = st.slider(
            "Number of synthetic questions", 5, 40, EVAL_NUM_QUESTIONS,
            help="More questions = more accurate eval, but slower and more API calls."
        )
    with col2:
        st.metric("Est. Groq calls", n_questions // 2 + n_questions)

    # Check if dataset already exists on disk
    dataset_exists = os.path.exists(EVAL_DATASET_PATH)
    if dataset_exists and st.session_state.eval_dataset is None:
        try:
            st.session_state.eval_dataset = DatasetGenerator.load(EVAL_DATASET_PATH)
        except Exception:
            pass

    if st.session_state.eval_dataset:
        st.success(
            f"✅ Dataset ready: {len(st.session_state.eval_dataset)} questions loaded."
        )
        with st.expander("👀 Preview dataset"):
            for i, item in enumerate(st.session_state.eval_dataset[:3]):
                st.markdown(f"**Q{i+1}:** {item['question']}")
                st.caption(f"Ground truth: {item['ground_truth'][:200]}…")
                st.divider()

    gen_btn_label = "🔄 Regenerate dataset" if st.session_state.eval_dataset else "⚡ Generate dataset"
    if st.button(gen_btn_label):
        generator = DatasetGenerator(
            vector_store=rag.vector_store,
            llm=rag.llm,
            rag_system=rag,
        )
        progress = st.progress(0, text="Generating Q&A pairs…")
        try:
            with st.spinner("Calling Groq to generate questions…"):
                dataset = generator.generate(
                    num_questions=n_questions,
                    save_path=EVAL_DATASET_PATH,
                )
            st.session_state.eval_dataset = dataset
            progress.progress(100, text="Done!")
            st.success(f"✅ Generated {len(dataset)} questions.")
            st.rerun()
        except Exception as e:
            st.error(f"Generation failed: {e}")

    st.divider()

    # ---- Step 2: Run evaluation ----
    st.subheader("Step 2 — Run RAGAS Evaluation")

    if not st.session_state.eval_dataset:
        st.info("Generate a dataset above first.")
    else:
        judge_llm_choice = st.radio(
            "Judge LLM",
            ["Groq (primary)", "Google Gemini Flash (fallback)"],
            horizontal=True,
            help="Groq is free but rate-limited. Gemini Flash is free with a Google AI Studio key."
        )
        force_gemini = judge_llm_choice.startswith("Google")

        if force_gemini:
            gemini_key = st.text_input(
                "Google AI Studio API key",
                type="password",
                help="Get one free at https://aistudio.google.com/app/apikey"
            )
            if gemini_key:
                os.environ["GOOGLE_API_KEY"] = gemini_key

        eval_mode = st.radio(
            "Evaluation mode",
            ["Current system (hybrid + reranker)", "Baseline (dense only)", "Both — show before/after"],
            horizontal=True,
        )

        if st.button("🚀 Run evaluation"):
            evaluator = RAGEvaluator(prefer_groq=not force_gemini)
            dataset   = st.session_state.eval_dataset

            # Helper: generate answers using a specific RAG config
            def _generate_answers(use_hybrid: bool, use_reranker: bool) -> List[dict]:
                rag.use_hybrid = use_hybrid
                rag._hybrid_retriever.use_reranker = use_reranker
                samples = []
                for item in dataset:
                    try:
                        result = rag.ask(item["question"], top_k=top_k, score_threshold=threshold)
                        samples.append({
                            "question":     item["question"],
                            "answer":       result["answer"],
                            "contexts":     item.get("contexts", [d["content"] for d in result["chunks"]]),
                            "ground_truth": item["ground_truth"],
                        })
                    except Exception as e:
                        st.warning(f"Skipped question: {e}")
                return samples

            try:
                if eval_mode in ["Current system (hybrid + reranker)", "Both — show before/after"]:
                    with st.spinner("Generating answers (hybrid + reranker)…"):
                        samples_hybrid = _generate_answers(use_hybrid=True, use_reranker=True)
                    with st.spinner("Running RAGAS on hybrid results…"):
                        results_hybrid = evaluator.evaluate(
                            samples_hybrid,
                            save_path=EVAL_RESULTS_PATH,
                            force_gemini=force_gemini,
                        )
                    st.session_state.eval_results = results_hybrid

                if eval_mode in ["Baseline (dense only)", "Both — show before/after"]:
                    with st.spinner("Generating answers (dense only, no reranker)…"):
                        samples_baseline = _generate_answers(use_hybrid=False, use_reranker=False)
                    with st.spinner("Running RAGAS on baseline results…"):
                        results_baseline = evaluator.evaluate(
                            samples_baseline,
                            save_path=EVAL_RESULTS_PATH.replace(".json", "_baseline.json"),
                            force_gemini=force_gemini,
                        )
                    st.session_state.baseline_results = results_baseline

                # Restore original toggles
                rag.use_hybrid = use_hybrid
                rag._hybrid_retriever.use_reranker = use_reranker

                st.success("✅ Evaluation complete!")
                st.rerun()

            except Exception as e:
                st.error(f"Evaluation failed: {e}")
                if "GOOGLE_API_KEY" not in os.environ and force_gemini:
                    st.info("Tip: Make sure you entered your Google AI Studio key above.")

    st.divider()

    # ---- Step 3: Display results ----
    st.subheader("Step 3 — Results")

    METRIC_LABELS = {
        "faithfulness":      "Faithfulness",
        "answer_relevancy":  "Answer Relevancy",
        "context_precision": "Context Precision",
        "context_recall":    "Context Recall",
    }
    METRIC_HELP = {
        "faithfulness":      "Are claims in the answer supported by retrieved context? (Higher = less hallucination)",
        "answer_relevancy":  "Is the answer relevant to the question?",
        "context_precision": "Are the top retrieved chunks actually useful for answering?",
        "context_recall":    "Did retrieval capture all information needed to answer?",
    }

    def _show_metric_row(results: dict, label: str):
        agg = results.get("aggregate", {})
        cols = st.columns(4)
        for col, (key, display) in zip(cols, METRIC_LABELS.items()):
            val = agg.get(key)
            col.metric(
                label=display,
                value=f"{val:.2%}" if val is not None else "N/A",
                help=METRIC_HELP[key],
            )

    def _delta_color(delta: float) -> str:
        return "normal" if delta >= 0 else "inverse"

    has_hybrid   = st.session_state.eval_results is not None
    has_baseline = st.session_state.baseline_results is not None

    if not has_hybrid and not has_baseline:
        st.info("Run an evaluation above to see results here.")

    if has_hybrid and not has_baseline:
        st.markdown("#### 🏅 Hybrid + Reranker")
        _show_metric_row(st.session_state.eval_results, "hybrid")

    if has_baseline and not has_hybrid:
        st.markdown("#### 📋 Baseline (Dense Only)")
        _show_metric_row(st.session_state.baseline_results, "baseline")

    # Before / After comparison
    if has_hybrid and has_baseline:
        st.markdown("#### 📈 Before vs After — Hybrid Retrieval + Cross-Encoder Reranking")

        baseline_agg = st.session_state.baseline_results.get("aggregate", {})
        hybrid_agg   = st.session_state.eval_results.get("aggregate", {})

        cols = st.columns(4)
        for col, (key, display) in zip(cols, METRIC_LABELS.items()):
            base_val   = baseline_agg.get(key)
            hybrid_val = hybrid_agg.get(key)
            if base_val is not None and hybrid_val is not None:
                delta = hybrid_val - base_val
                col.metric(
                    label=display,
                    value=f"{hybrid_val:.2%}",
                    delta=f"{delta:+.2%}",
                    delta_color=_delta_color(delta),
                    help=METRIC_HELP[key],
                )
            else:
                col.metric(label=display, value="N/A")

        st.caption(
            f"Baseline: {st.session_state.baseline_results.get('num_samples','?')} samples  |  "
            f"Hybrid: {st.session_state.eval_results.get('num_samples','?')} samples  |  "
            f"Evaluated: {st.session_state.eval_results.get('timestamp','')}"
        )

        # Per-sample table (collapsible)
        with st.expander("📋 Per-sample breakdown (hybrid)"):
            per_sample = st.session_state.eval_results.get("per_sample", [])
            if per_sample:
                import pandas as pd
                df = pd.DataFrame(per_sample)[
                    ["question"] + [k for k in METRIC_LABELS if k in per_sample[0]]
                ]
                # Format floats as percentages
                for col_name in METRIC_LABELS:
                    if col_name in df.columns:
                        df[col_name] = df[col_name].map(lambda x: f"{x:.2%}" if x == x else "N/A")
                st.dataframe(df, use_container_width=True)

    # Download buttons
    if has_hybrid or has_baseline:
        st.markdown("---")
        dl_col1, dl_col2, dl_col3 = st.columns(3)

        if has_hybrid:
            with dl_col1:
                st.download_button(
                    "⬇️ Download hybrid results (JSON)",
                    data=json.dumps(st.session_state.eval_results, indent=2),
                    file_name="ragas_hybrid_results.json",
                    mime="application/json",
                )
        if has_baseline:
            with dl_col2:
                st.download_button(
                    "⬇️ Download baseline results (JSON)",
                    data=json.dumps(st.session_state.baseline_results, indent=2),
                    file_name="ragas_baseline_results.json",
                    mime="application/json",
                )
        if st.session_state.eval_dataset:
            with dl_col3:
                st.download_button(
                    "⬇️ Download eval dataset (JSON)",
                    data=json.dumps(st.session_state.eval_dataset, indent=2),
                    file_name="eval_dataset.json",
                    mime="application/json",
                )

