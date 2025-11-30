## RAG Service (FastAPI + Ollama + Gemini)

A fast Retrieval-Augmented Generation (RAG) microservice:
- Embeddings: `nomic-embed-text` (via Ollama, local)
- Generation: `gemini-1.5-flash` (via Google Generative AI, very fast)

### Prerequisites
- Ollama with models:
```bash
ollama pull nomic-embed-text
```
- Google Generative AI API key (set `GOOGLE_API_KEY`)

### Setup
```bash
cd rag-service
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
Create `.env` (recommended):
```
EMBEDDING_MODEL=nomic-embed-text
GENERATION_MODEL=gemini-1.5-flash
INDEX_DIR=data
PORT=8088
GOOGLE_API_KEY=your_key
```

### Run
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8088 --reload
```

### API
- POST `/ingest` – JSON docs
```bash
curl -X POST http://localhost:8088/ingest -H 'Content-Type: application/json' -d '{
  "documents": [
    {"id":"guide1","text":"Timely planting for maize in Kenya ..."}
  ]
}'
```

- POST `/ingest/files` – Text and PDF files (PDFs parsed server-side)
```bash
curl -X POST http://localhost:8088/ingest/files \
  -F "files=@/path/maize_best_practices.md" \
  -F "files=@/path/pests_armyworm.pdf"
```

- POST `/query`
```bash
curl -X POST http://localhost:8088/query -H 'Content-Type: application/json' -d '{
  "question": "How do I manage fall armyworm?",
  "k": 4
}'
```

### Notes
- Similarity: cosine via normalized vectors with FAISS IP index.
- For large PDFs, consider chunking upstream for better retrieval granularity.
- Swap models via env vars. 