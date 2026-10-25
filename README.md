# SchemeSathi 🇮🇳

An English-language, RAG-based Streamlit chatbot that helps users explore Indian government schemes. It is designed as a resume/portfolio project: it shows the retrieval layer, grounded answer flow, source links, chat history, and a graceful offline fallback.

## Features

- Retrieves relevant schemes from a curated JSON knowledge base using local keyword scoring
- Generates answers only from retrieved context when `GEMINI_API_KEY` is configured
- Works without an API key using transparent extractive responses
- Shows official source links with every response
- Includes a clear informational disclaimer and avoids definitive eligibility claims

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

To enable Gemini-generated answers, copy `.env.example` to `.env` and add your Gemini API key. `python-dotenv` loads it automatically when the app starts. The app still works without a key.

## Architecture

```text
User question → local retrieval → top scheme records → grounded LLM (optional) → answer + official citations
                                                └── local grounded fallback (no API key)
```

## Build the local vector store

The corpus has been cleaned, enriched with metadata, and chunked. Build the semantic
retrieval store from the recursive chunks with:

```bash
python -m src.embeddings.build_faiss_index
```

This downloads `BAAI/bge-base-en-v1.5` on first run into `data/models/` and saves the
normalized-vector FAISS index, chunk payload, and a reproducibility manifest in
`data/vector_store/bge_base_en_v1_5/`. These generated artifacts are ignored by Git.

## Evaluate retrieval

Run the difficult, document-level retrieval suite after creating the vector store:

```bash
python -m src.embeddings.evaluate_retrieval
```

It uses 100 paraphrased and disambiguating questions, evaluates deduplicated scheme
results (not repeated chunks), and reports Precision@1/3/5, Recall@1/3/5, and MRR@5.
Each case records its gold-relevant-document count, retrieved-document count, and
relevant-retrieved-document count at each cutoff. The full results are saved to
`data/evaluation/results/retrieval_report.json`.

## Inspect retrieved chunks

Open `notebooks/retrieval_demo.ipynb` in Jupyter and run its cells to see the exact
chunks returned for an eligibility question. Change `QUESTION` in the notebook to
inspect another user question.

## Data and scope

`data/schemes.json` is a small, curated demo corpus of Central schemes. Each record includes a link to an official source. Government programme details can change, so users should verify eligibility, benefits, documents, and application steps on the linked portal before applying.

## Portfolio talking points

- Designed a RAG workflow with source-aware retrieval and hallucination controls
- Built a user-friendly Streamlit chat interface with persistent session history
- Implemented resilient LLM integration with an offline retrieval fallback
- Made trust visible through official citations and eligibility disclaimers
