# GenAI RAG Backend

Backend service for domain-aware Retrieval-Augmented Generation (RAG) with:
- FastAPI API layer
- Azure AI Search retrieval (docs, code, PTR)
- Azure OpenAI embeddings + chat completion
- Specialized PTR summarize and similar-PTR investigation flows

## Current Project Structure

```text
/Users/kumarn/pwspace/genai_rag_backend
├── api/
│   └── app.py
├── pipeline/
│   └── rag_pipeline.py
├── services/
│   ├── classifier.py
│   ├── ptr_analysis.py
│   ├── repo_info.py
│   └── similar_ptr_analysis.py
├── retrieval/
│   └── retrievers.py
├── prompts/
│   └── prompts.py
├── core/
│   ├── logging_config.py
│   └── utils.py
├── data/
│   ├── domain_terms.yaml
│   └── workflows.yaml
└── requirements.txt
```


## API Endpoints

### `POST /ask`
Request body:
```json
{
  "query": "your question",
  "history": [{"role": "user", "content": "..."}]
}
```

Response (current behavior):
```json
{
  "answer": "...",
  "sources": []
}
```

Important:
- Responses always include references/citations in the answer content irrespective of query type.

### `GET /health`
Returns:
```json
{"status": "ok"}
```

## Runtime Flow (Current Logic)

Main orchestration is in `pipeline/rag_pipeline.py`:
1. Chit-chat short-circuit via `core/utils.py`.
2. PTR summary intent check (`services/ptr_analysis.py`):
   - Trigger examples: `summarize ptr 1234567`, `ptr #1234567 summary`.
   - Fetches PTR details from MCP, enriches with docs context, returns markdown answer + sources.
3. Similar PTR intent check (`services/similar_ptr_analysis.py`):
   - Trigger pattern: `find similar ptrs <issue text>`.
   - Searches PTR index + docs enrichment, returns answer + sources.
4. Standard RAG path:
   - Classify + rewrite query (`services/classifier.py`) into `docs_question` or `mixed`.
   - Embed rewritten query.
   - Retrieve docs-only or docs+code (`retrieval/retrievers.py`, threaded mixed retrieval).
   - Build final prompt using templates in `prompts/prompts.py` + repo/domain/workflow context.
   - Invoke Azure OpenAI chat model and return answer.

## Core Components

- `api/app.py`: FastAPI app, CORS, `/ask`, `/health`.
- `pipeline/rag_pipeline.py`: central orchestration and client initialization.
- `services/classifier.py`: LLM-based classification + query rewrite + repo inference.
- `services/ptr_analysis.py`: PTR summarize mode (MCP tool flow + doc planning).
- `services/similar_ptr_analysis.py`: similar PTR investigation mode.
- `retrieval/retrievers.py`: Azure AI Search retrieval/filtering/sorting for docs/code/PTR.
- `prompts/prompts.py`: classifier/system prompts.
- `core/logging_config.py`: logger setup.
- `data/*.yaml`: domain and workflow context injected into prompts.

## Environment Variables

.env reference:
- `.env.example` is provided as a reference template with dummy values.

Required (core RAG path):
- `AZURE_SEARCH_ENDPOINT`
- `AZURE_SEARCH_API_KEY`
- `INDEX_DOCS`
- `INDEX_CODE`
- `TOP_AISEARCH_DOC_RESULTS`
- `TOP_AISEARCH_CODE_RESULTS`
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_KEY`
- `AZURE_OPENAI_EMBEDDING_DEPLOYMENT`
- `AZURE_OPENAI_CHAT_DEPLOYMENT`
- `AZURE_OPENAI_API_VERSION`

Optional / feature-specific:
- `INDEX_PTRS`
- `TOP_AISEARCH_CODE_RESULTS_PER_REPO`
- `TOP_AISEARCH_PTR_RESULTS`
- `TOP_AISEARCH_PTR_KNN`
- `K_TOP_PTRS`
- `PTR_SEMANTIC_CONFIG`
- `PTR_VECTOR_FIELD`
- `PTR_MAX_DOCS_TOTAL`
- `PTR_SIMILAR_MAX_PTRS_CONTEXT`
- `MCP_URL`
- `MCP_USER`
- `MCP_PASS`
- `MCP_EMAIL`

## Local Run

If venv is already active:
```bash
python -m uvicorn api.app:app --reload
```

Or without activating venv:
```bash
/Users/kumarn/pwspace/genai_rag_backend/venv/bin/python -m uvicorn api.app:app --reload
```

## Known Current Behaviors

- CORS is open (`allow_origins=["*"]`).
- Chit-chat detection uses simple keyword matching.
- Query history usage is limited to last 3 conversation turns.
- Domain and workflow YAML files are loaded per request in the pipeline.
- `services/ptr_analysis.py` currently contains a temporary SSL verification workaround for MCP calls.
