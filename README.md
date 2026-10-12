# AI Research Assistant (RAG MVP)

Upload TXT/PDF documents, ask questions, get answers with source citations.

## Run

```bash
source .venv/bin/activate
pip install -r requirements.txt   # already done if you followed setup
streamlit run src/app.py
```

Without an API key the app runs in **retrieval-only mode** (free, no API calls).

## Enable LLM answers (OpenAI)

OpenAI API usage is billed separately from a ChatGPT subscription.

```bash
cp .env.example .env
# edit .env and set OPENAI_API_KEY=sk-...
```

Restart Streamlit after editing `.env`. `.env` is listed in `.gitignore`.

## Files

| File | Role |
|---|---|
| `src/app.py` | Streamlit UI, wires the pipeline together |
| `src/loaders.py` | TXT/PDF → text, one entry per page |
| `src/chunking.py` | Overlapping word-window chunks |
| `src/retrieval.py` | BM25 keyword search (pure Python) |
| `src/llm.py` | Grounded prompt, OpenAI call, "not enough info" detection |
| `src/config.py` | Reads `.env` / environment variables |
