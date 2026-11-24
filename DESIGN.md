# Document Q&A System – Detailed Design Document

## 1. Overview

This project provides a document-based question answering (RAG) system built around a reusable document pipeline and exposed through a FastAPI service. Users upload PDF, TXT, or JSON documents, the system extracts and chunks their content, stores embeddings in a persistent Chroma collection, and answers natural-language questions using only the retrieved chunks for the specified uploads.

The repository also includes a command-line pipeline/ingestor for local indexing and querying, and a small standalone Gemini tool-calling demo (`main.py`) unrelated to the document Q&A API.

## 2. Goals and scope

### 2.1 Goals
- **Document ingestion:** Load supported document types (PDF, TXT, JSON) from files/directories or API uploads.
- **Chunking:** Split content into retrievable chunks suitable for embedding and search.
- **Vector indexing:** Embed and persist chunks in Chroma for later retrieval.
- **Retrieval-augmented answering:** Retrieve relevant chunks and generate answers grounded in indexed document content.
- **API access:** Provide HTTP endpoints to upload documents and ask questions.
- **Local/containerized execution:** Run locally via Python/Uvicorn and via Docker/Cloud Run (demo deployment).

### 2.2 In scope
- Upload validation, storage, indexing, and question answering over uploaded document sets (`upload_id`s).
- Local CLI pipeline execution over a document directory.
- Configurable persistence/upload directories via environment variables.
- Basic operational concerns documented for demo deployment (ephemeral storage, no app-level auth).

### 2.3 Out of scope / non-goals
- Durable production-grade storage guarantees on Cloud Run demo setup.
- Application-level user authentication/authorization and multi-tenant isolation beyond upload ID filtering.
- Document deletion/update APIs, background indexing jobs, or indexing status tracking.
- OCR for scanned PDFs.
- Production monitoring, quotas enforcement, backups, or horizontal scaling of shared state.

## 3. System context and architecture

### 3.1 High-level architecture

```mermaid
flowchart LR
    subgraph Clients
      U[User / API Client]
      CLI[CLI User]
    end

    subgraph App["Application"]
      API[FastAPI app\n(document_pipeline.api)]
      ING[CLI ingestor\n(document_pipeline.document_ingestor)]
      PIPE[DocumentPipeline]

      L[DocumentLoader]
      C[DocumentChunker]
      VS[ChromaVectorStore]
      R[DocumentRetriever]
      A[DocumentAgent]
    end

    subgraph External
      GEM_E[Gemini Embeddings\nmodels/gemini-embedding-001]
      GEM_C[Gemini Chat\n gemini-3.5-flash-lite]
    end

    subgraph Storage
      FS[Uploads directory\nUPLOAD_DIRECTORY]
      CH[(Chroma persistent store\nCHROMA_PERSIST_DIRECTORY / embedding_collection)]
    end

    U -->|POST /documents, /questions| API
    CLI --> ING --> PIPE
    API --> L --> C --> VS
    API --> R --> A
    PIPE --> L --> C --> VS --> R --> A

    VS <-->|embed/query vectors| GEM_E
    A <-->|generate answer + tool use| GEM_C
    API --> FS
    VS --> CH
```

### 3.2 Core design principles
- **Separation of concerns:** Loading, chunking, storage, retrieval, and agent reasoning are implemented as distinct classes.
- **Composability:** The same core components are reused by the API and CLI pipeline.
- **Grounded responses:** The agent is prompted to retrieve context via a tool and answer only from returned document content.
- **Deterministic chunk IDs:** Chunk IDs are derived from content/source metadata to make indexing stable for identical inputs.

## 4. Component design

Source package: `document_pipeline/`

### 4.1 Configuration (`config.py`)
Central constants and environment-driven settings:
- `DEFAULT_DOCUMENT_LOCATION = "resources"`
- `CHUNK_SIZE = 200`, `CHUNK_OVERLAP = 30`
- `EMBEDDING_MODEL = "models/gemini-embedding-001"`
- `CHROMA_COLLECTION_NAME = "embedding_collection"`
- `CHROMA_PERSIST_DIRECTORY` (env, default `chroma_store`)
- `DEFAULT_SEARCH_RESULT_COUNT = 4`
- `QUESTION` default prompt for CLI
- `GENERATION_MODEL = "gemini-3.5-flash-lite"`
- `REQUEST_TIMEOUT_SECONDS = 180`
- `get_gemini_api_key()`: loads `.env` via `python-dotenv` and requires `GEMINI_API_KEY`.

Relative Chroma paths are resolved against the current working directory at runtime.

### 4.2 Document loading (`loader.py` – `DocumentLoader`)
**Responsibility:** Discover and load supported files into LangChain `Document` objects with normalized metadata.

- **Input:** Directory path (default `resources`; relative paths resolved from CWD).
- **Supported types:** `.pdf`, `.txt`, `.json` (case-insensitive, recursive globbing).
- **Loaders:**
  - PDF: `PyPDFLoader` (one document per page).
  - TXT: `TextLoader` with UTF-8 and autodetected encoding fallback behavior.
  - JSON: `JSONLoader` (`jq_schema="."`, `text_content=False`) when available; falls back to `_StandardJsonLoader` (Python `json` module, pretty-printed JSON text) if `jq`/LangChain JSON loader dependency import fails.
- **Metadata enrichment:** Sets/normalizes `source` (absolute path string), `file_name`, and `file_type` (`pdf|txt|json`).
- **Ordering:** Documents sorted by `(source, page)` for deterministic processing.
- **Errors:** Raises `NotADirectoryError` if location is invalid.

### 4.3 Chunking (`chunker.py` – `DocumentChunker`)
**Responsibility:** Split loaded documents into chunks for embedding/retrieval.

- **Text/PDF:** `RecursiveCharacterTextSplitter` with `chunk_size` and `chunk_overlap` from config.
- **JSON:** `RecursiveJsonSplitter(max_chunk_size=chunk_size)`.
  - For dict/list JSON content: `split_text(..., convert_lists=True, ensure_ascii=False)`.
  - For scalar JSON values: wrapped as `{"root": <value>}` before splitting.
  - Each JSON chunk becomes a `Document` preserving original metadata.
- **State/reporting:** Stores chunks in `self.chunks` and can print chunk metadata/content.

### 4.4 Vector store management (`vector_store.py` – `ChromaVectorStore`)
**Responsibility:** Create/open a persistent Chroma collection, generate embeddings, and verify stored records.

- **Persistence:** `persist_directory` (env-configurable, resolved from CWD if relative), collection `embedding_collection`.
- **Embedding model:** `GoogleGenerativeAIEmbeddings(model=models/gemini-embedding-001, task_type="RETRIEVAL_DOCUMENT")`.
- **Chunk IDs:** SHA-256 hash of `(source, page, index, page_content)` joined with NUL separators; provides stable identifiers for inserted chunks.
- **Indexing:** `Chroma.from_documents(..., ids=document_ids, persist_directory=...)`.
- **Rate-limit handling:** Retries embedding/index creation on Gemini 429 / `RESOURCE_EXHAUSTED` up to 3 attempts.
  - Uses provider-suggested `retryDelay` when parseable and <= 120s; otherwise exponential backoff starting at 5s (capped at 120s).
  - If suggested delay exceeds cap or attempts exhausted, raises.
- **Verification/debug utilities:** `open_vector_store()`, `print_embeddings()`, `print_stored_documents()` (compares stored content+metadata against expected chunks).
- **Note:** `create_vector_store` writes provided chunks with deterministic IDs; it does not implement deduplication across separate upload requests beyond ID collisions for identical chunk identities.

### 4.5 Retrieval (`retriever.py` – `DocumentRetriever`)
**Responsibility:** Query the vector store and format results for agent/tool consumption and CLI inspection.

- **Construction:** Accepts a `Chroma` store, `k` (default 4), and optional `metadata_filter` (e.g. `{"upload_id": {"$in": [...]}}`), exposed as a LangChain retriever via `vector_store.as_retriever(search_kwargs=...)`.
- **Tool-facing retrieval:** `retrieve_documents(query)` returns a formatted string containing source, full metadata JSON, and content for each hit, or `"No relevant documents were found."` Validates non-empty query and presence of store.
- **CLI/debug retrieval:** `print_vector_store_results(query, k)` performs similarity search with scores and prints distance/metadata/content.

### 4.6 Agent (`agent.py` – `DocumentAgent`)
**Responsibility:** Answer questions using an LLM agent constrained to retrieved document context.

- **LLM:** `ChatGoogleGenerativeAI` (`gemini-3.5-flash-lite`, timeout 180s, API key from config).
- **Tooling:** Exposes `retrieve_documents` as a `StructuredTool` backed by `DocumentRetriever.retrieve_documents`.
- **System behavior:** Agent is instructed to call `retrieve_documents` before answering, answer only from returned content, and state when documents do not contain the answer.
- **Execution:** `create_agent()` builds a LangGraph/`langchain.agents` compiled agent; `answer(prompt)` invokes it and returns the final `AIMessage` text; `run(prompt)` prints the answer for CLI use.
- **Validation:** Empty prompts rejected; answering requires an available vector store.

### 4.7 Pipeline orchestration (`pipeline.py` – `DocumentPipeline`)
**Responsibility:** End-to-end CLI/local workflow composition.

`run(prompt, search_query=None)` performs:
1. Load documents from configured directory and print report.
2. Split into chunks and print chunks.
3. Create/persist Chroma vector store, verify stored documents, print embeddings.
4. Create retriever (and optionally print similarity search results).
5. Create and run `DocumentAgent` with the prompt.

Returns the produced chunks. Intended for interactive/local inspection; verbose by design.

### 4.8 CLI entrypoint (`document_ingestor.py`)
Prompts for a question (default from config) and runs `DocumentPipeline().run(...)`.

### 4.9 Standalone demo (`main.py`)
Separate async CLI demo using `langchain.agents.create_agent` with a local `hello_world` tool and Gemini chat streaming. Not part of the document ingestion/retrieval API.

## 5. API design (`api.py`)

Framework: FastAPI (`title="Document Q&A API"`).

### 5.1 Configuration/constants
- `UPLOAD_DIRECTORY` (env, default `uploads`), resolved to absolute path.
- Allowed file types: `.pdf`, `.txt`, `.json`.
- Limits: max 10 files/request, 20 MiB/file, 50 MiB total/request.
- Question constraints: non-blank, 1–4000 chars; `upload_ids`: 1–10 items.

### 5.2 Models
- **`QuestionRequest`**: `question: str`, `upload_ids: list[str]` (validated/trimmed).
- **`UploadResponse`**: `upload_id`, `files[]`, `chunks_indexed`.
- **`QuestionResponse`**: `answer`.

### 5.3 Endpoints

| Method | Path | Purpose | Success | Key errors |
|---|---|---|---|---|
| GET | `/health` | Liveness/health check | `200 {"status":"ok"}` | — |
| POST | `/documents` | Upload and index documents (`multipart/form-data`, field `files`) | `201 UploadResponse` | `400` no files; `413` too many/large; `415` unsupported type; `422` empty/unreadable/no text |
| POST | `/questions` | Answer question over specified uploads (`application/json`) | `200 QuestionResponse` | `404` no indexed docs for supplied upload IDs; validation errors for bad request body; runtime errors if dependencies/LLM unavailable |

### 5.4 Upload/index flow
1. Validate request size/count/types and read file bytes with per-file cap checks.
2. Generate `upload_id` (`uuid4().hex`) and ensure upload root exists.
3. Stage uploaded files in a temporary directory under the upload root (`{index:02d}-{uuid}{suffix}`).
4. Load documents from staging via `DocumentLoader`.
5. Rewrite each document’s metadata to final paths/names and attach `upload_id`, `file_name`, `original_filename`, `source`.
6. Chunk via `DocumentChunker`; reject if no chunks produced.
7. Index chunks in Chroma via `ChromaVectorStore().create_vector_store(chunks)`.
8. Rename staging directory to final `UPLOAD_DIRECTORY/<upload_id>` so stored `source` metadata points to persisted files.
9. Return `upload_id`, original filenames, and chunk count.

### 5.5 Question flow
1. Open Chroma store and build metadata filter `{"upload_id": {"$in": request.upload_ids}}`.
2. Verify at least one stored document exists for the filter (`get(where=..., limit=1)`); otherwise 404.
3. Create `DocumentRetriever(vector_store, metadata_filter=...)` and `DocumentAgent(retriever)`.
4. Generate and return the answer.

Answers are scoped to chunks matching the provided upload IDs at retrieval time; there is no user identity/authentication layer in the application.

## 6. Data model and metadata

### 6.1 Document metadata (typical fields)
- **From loader:** `source`, `file_name`, `file_type`, and loader-specific fields (e.g. PDF `page`).
- **From API upload enrichment:** `upload_id`, `original_filename`, stored `source` path, normalized `file_name`.

Metadata is stored with chunks in Chroma and returned in retrieval/tool output to support traceability.

### 6.2 Chunk identity and persistence
- Vector IDs are deterministic hashes of source/page/index/content.
- Chroma data persists under `CHROMA_PERSIST_DIRECTORY` (collection `embedding_collection`).
- API uploads are stored under `UPLOAD_DIRECTORY/<upload_id>/` after successful indexing.

## 7. External dependencies and integrations

### 7.1 Runtime dependencies (from `requirements.txt`)
- LangChain ecosystem: `langchain`, `langchain-chroma`, `langchain-community`, `langchain-google-genai`, `langchain-text-splitters`, `langgraph` (transitively via agent APIs).
- Document/utility: `pypdf`, `jq`, `python-dotenv`.
- API/server: `fastapi`, `python-multipart`, `uvicorn`.

### 7.2 External services
- **Google Gemini API**: required for embeddings and chat generation (`GEMINI_API_KEY`). Subject to network availability, quota/rate limits, and cost.

## 8. Deployment and runtime configuration

### 8.1 Environment variables
- `GEMINI_API_KEY` (required for embedding/answering operations).
- `CHROMA_PERSIST_DIRECTORY` (default `chroma_store`, or `/tmp/chroma_store` in Dockerfile).
- `UPLOAD_DIRECTORY` (default `uploads`, or `/tmp/uploads` in Dockerfile).
- `PORT` (container runtime; Dockerfile defaults to `8080`).

### 8.2 Local execution
- **API:** `uvicorn document_pipeline.api:app --reload` (default local docs at `http://127.0.0.1:8000/docs` per README) or containerized on port 8080.
- **CLI:** `python -m document_pipeline.document_ingestor` or `DocumentPipeline` usage.

### 8.3 Docker
- Base image `python:3.12-slim`, installs requirements, copies source, runs as non-root `appuser` (UID 10001), exposes 8080.
- Default command: `uvicorn document_pipeline.api:app --host 0.0.0.0 --port ${PORT:-8080}`.
- `.dockerignore` excludes virtual envs, caches, local stores/uploads, `.git`, etc.

### 8.4 Cloud Run demo considerations
- Demo guidance uses `/tmp` for writable storage, single instance (`--max 1`, `--concurrency 1`) for best-effort affinity; **data is ephemeral** across restarts/scale-to-zero/new instances.
- Secrets (e.g. Gemini key) should be provided via Secret Manager, not baked into images/source.
- README recommends keeping service private/IAM-authenticated for demo unless additional protections are added.

## 9. Error handling, validation, and reliability

- **Input validation:** Strict upload type/size/count checks and question/upload ID validation at API boundary.
- **Operational safeguards:** Empty/unreadable documents and empty chunk results return client errors (422).
- **Retrieval scoping checks:** Question requests fail with 404 if no indexed chunks exist for requested upload IDs.
- **Embedding retries:** Bounded retries for Gemini rate-limit conditions with delay handling/capping.
- **JSON loader resilience:** Fallback loader avoids hard dependency failure when `jq` is unavailable.
- **Known tradeoffs:** No transactional rollback across filesystem + vector-store writes beyond staging/rename sequencing; partial persistence semantics depend on failure point.

## 10. Security and privacy considerations

- **Secrets:** `GEMINI_API_KEY` must not be committed or embedded in images; use `.env` locally or secret manager in cloud.
- **Authentication/authorization:** No application-level auth; upload IDs are not access control tokens. Deployments should rely on platform controls (e.g. Cloud Run IAM) or add API auth/rate limiting before public exposure.
- **Data handling:** Uploaded files and derived chunks/embeddings are stored locally/in configured directories; treat contents as sensitive as applicable.
- **Resource controls:** Request-level size/count limits exist, but no global quota, retention, or abuse protection mechanisms are implemented.

## 11. Testing

- Tests located under `tests/` (e.g. `test_api.py`, `test_pipeline_components.py`).
- README states tests cover API upload/question behavior, chunking, retrieval formatting, and agent responses with mocked Gemini/vector-store operations (no live API key/quota required).
- Typical run: `python -m unittest discover -s tests -v` (or venv-specific interpreter).

## 12. Limitations and future improvements

### 12.1 Current limitations
- Ephemeral storage in demo container/Cloud Run `/tmp` setups; no durable managed document/vector persistence by default.
- No delete/list/manage APIs for uploads or indexed content.
- Single-process/simple deployment assumptions; shared-state scaling not addressed.
- No OCR; limited to text-extractable PDFs and text/JSON inputs.
- Minimal observability (no structured tracing/metrics) and no explicit deduplication policy across repeated uploads of identical content.

### 12.2 Potential enhancements
- Add durable object storage + managed vector DB for production persistence.
- Implement document lifecycle APIs (list/delete/reindex/status) and metadata-based governance.
- Add authentication/authorization, rate limiting, and audit logging.
- Introduce background ingestion jobs and progress reporting for large uploads.
- Improve retrieval (hybrid search, reranking, citations/snippet offsets) and answer evaluation.
- Expand supported formats and OCR pipeline for scanned documents.
- Make chunking/embedding/retrieval parameters configurable per request or workload.

## 13. Key source files

- API: `document_pipeline/api.py`
- Orchestration/pipeline: `document_pipeline/pipeline.py`, `document_pipeline/document_ingestor.py`
- Core components: `document_pipeline/loader.py`, `document_pipeline/chunker.py`, `document_pipeline/vector_store.py`, `document_pipeline/retriever.py`, `document_pipeline/agent.py`
- Config: `document_pipeline/config.py`, `.env.example`
- Containerization: `Dockerfile`, `.dockerignore`
- Standalone demo: `main.py`
- Documentation: `README.md`, `DESIGN.md`
