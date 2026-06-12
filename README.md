# QueryBased_PDF_Summarizer
# 📚 QueryBased PDF Summarizer

An intelligent research paper question-answering system built on a **Retrieval-Augmented Generation (RAG)** architecture. Upload multiple PDF research papers, ask questions in plain English, and get detailed answers with **exact page and source citations** — including insights from figures and charts.


---

## 🎯 What it does

- Upload one or more research papers in PDF format
- Ask any natural language question about the content
- Get a detailed, cited answer referencing the exact **page number** and **source file**
- Chat history is preserved across queries in the same session

---

## 🖥️ Deployed Link
https://querybasedpdfsummarizer.streamlit.app/


```
Q: What are the key characteristics of Agentic AI systems?

A: Agentic AI systems exhibit the following key characteristics:

• Adaptability — the ability to operate dynamically in evolving environments (Page 2, Agentic_AI_Survey.pdf)
• Autonomous decision-making — advanced reasoning with minimal human oversight (Page 3, Agentic_AI_Survey.pdf)
• Self-sufficiency — pursuing complex goals without structured instructions (Page 2, Agentic_AI_Survey.pdf)
```

---

## 🏗️ Architecture

```
PDF Upload
    │
    ▼
PDFProcessor (PyMuPDF)
    ├── Text extraction per page
    └── Image extraction + Vision LLM description
    │
    ▼
EmbeddingManager (all-MiniLM-L6-v2)
    │
    ▼
VectorStore (ChromaDB — cosine similarity)
    │
    ▼
User Query ──► Embed ──► Retrieve top-K chunks ──► Groq LLM ──► Cited Answer
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| PDF Processing | PyMuPDF (fitz) |
| Image Description | Llama 4 Scout Vision via Groq API |
| Embeddings | all-MiniLM-L6-v2 (SentenceTransformers) |
| Vector Store | ChromaDB (persistent, cosine similarity) |
| LLM | Llama 4 Scout via Groq API |
| Web Interface | Streamlit |
| Language | Python 3.10+ |

---

## 📁 Project Structure

```
QueryBased_PDF_Summarizer/
├── app.py                  # Streamlit web application
├── main.py                 # CLI ingestion script
├── config.py               # Centralised configuration
├── src/
│   ├── data_loader.py      # PDF text + image extraction, chunking
│   ├── embedding.py        # Sentence transformer embedding manager
│   ├── vectorstore.py      # ChromaDB vector store
│   ├── rag_system.py       # Retrieval + answer generation
│   └── image_processor.py  # Vision LLM image description
├── data/
│   ├── pdf_files/          # Uploaded PDFs (gitignored)
│   └── vector_store/       # ChromaDB index (gitignored)
├── requirements.txt
└── .gitignore
```

---

## ⚡ Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/deepti-9/QueryBased_PDF_Summarizer.git
cd QueryBased_PDF_Summarizer
```

### 2. Create a virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# or
source .venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up your API keys

Create a `.env` file in the project root:
```env
GROQ_API_KEY=your_groq_api_key_here
```

Create `.streamlit/secrets.toml`:
```toml
HF_TOKEN = "your_huggingface_token_here"
GROQ_API_KEY = "your_groq_api_key_here"
```

Get your keys here:
- **Groq API key:** https://console.groq.com
- **HuggingFace token:** https://huggingface.co/settings/tokens

### 5. Run the app
```bash
uv run streamlit run app.py
```

---

## 🔧 Configuration

All settings are in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `LLM_MODEL` | `llama-4-scout-17b-16e-instruct` | LLM for answer generation and vision |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer for embeddings |
| `CHUNK_SIZE` | `1000` | Characters per text chunk |
| `CHUNK_OVERLAP` | `150` | Overlap between chunks |
| `TOP_K_DEFAULT` | `5` | Number of chunks retrieved per query |
| `SCORE_THRESHOLD` | `0.35` | Minimum similarity score for retrieval |

---

## 📖 How to Use

1. **Open the app** at `(https://querybasedpdfsummarizer.streamlit.app/)`
2. **Upload PDFs** using the sidebar file uploader
3. **Click "Process & Index Documents"** — wait for ingestion to complete
4. **Type your question** in the chat input
5. **View the answer** with page citations and expandable source chunks
6. **Clear and re-upload** using the "Clear All Documents" button for a fresh session

---

## ✨ Features

- **Multi-document support** — query across multiple papers simultaneously
- **Multimodal indexing** — figures, charts, and diagrams described and searchable
- **Exact citations** — every answer cites page number and source filename
- **Persistent vector store** — documents survive app restarts without re-indexing
- **Chat history** — full conversation preserved within a session
- **Source viewer** — see the exact text/image chunks the LLM used
- **Configurable retrieval** — adjust Top-K and similarity threshold from the UI
- **GPU acceleration** — automatically uses CUDA if available

---

## 🔑 API Keys Required

| Service | Purpose | Get it at |
|---|---|---|
| Groq | LLM inference + vision description | https://console.groq.com |
| HuggingFace | Model download (all-MiniLM-L6-v2) | https://huggingface.co/settings/tokens |

---

## 📋 Requirements

```
streamlit>=1.23.0
langchain
langchain-groq
langchain-text-splitters
sentence-transformers
chromadb
pymupdf
pillow
python-dotenv
numpy
torch
```

---


## ⚠️ Limitations

- Scanned PDFs (images of text) require OCR pre-processing
- Author extraction optimised for IEEE format papers
- Image processing adds 1–3 seconds per figure during ingestion
- English language documents only

---

## 👨‍💻 Author

**Deepti Yadav**
B.Tech Computer Science & Engineering — 3rd Year
IGDTUW


---


## 🙏 Acknowledgements
- [Groq](https://groq.com) for low-latency LLM inference
- [ChromaDB](https://www.trychroma.com) for the vector store
- [Hugging Face](https://huggingface.co) for the sentence transformer model
- Lewis et al. (2020) — Retrieval-Augmented Generation paper

