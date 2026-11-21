# edureka_capstone

A small command-line LLM agent built with LangChain and Google's Gemini API. It
can call a local `hello_world` tool; ask it to use the tool to see the complete
flow.

## Setup (Windows PowerShell)

1. Create and activate a virtual environment:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

3. Create your local environment file and add your Gemini API key:

   ```powershell
   Copy-Item .env.example .env
   ```

   Replace the placeholder in `.env` with your key. Do not commit or share this
   file.

4. Run the agent:

   ```powershell
   python main.py
   ```

Try: `Use your hello_world tool and tell me what it returned.`

## Local document RAG pipeline

The `document_pipeline` package separates the workflow into focused classes:

- `DocumentLoader` loads PDF, JSON, and TXT files and reports their metadata.
- `DocumentChunker` splits text and JSON into chunks.
- `ChromaVectorStore` embeds and persists chunks in the `embedding_collection`
  collection under `chroma_store`, verifies stored records, and prints vectors.
- `DocumentRetriever` searches Chroma and formats retrieved chunks for a tool.
- `DocumentAgent` wraps the retriever method as a LangChain tool and answers
  prompts using the retrieved context.
- `DocumentPipeline` composes the classes for an end-to-end run.

The pipeline uses `GEMINI_API_KEY` from `.env` with Google's
`models/gemini-embedding-001` embedding model. Gemini 429 rate limits are
retried up to two times, honoring the suggested delay (up to 120 seconds) or
using 5- and 10-second backoff delays. The default question is
`summarise the opportunity cost`.

Run the complete workflow:

```powershell
python -m document_pipeline.document_ingestor
```

The script prompts for a question (press Enter to use the default). The agent
calls the `retrieve_documents` tool before answering.

## Document Q&A API

Start the API from the project root:

```powershell
uvicorn document_pipeline.api:app --reload
```

Open `http://127.0.0.1:8000/docs` for the interactive API. Upload one or more
PDF, TXT, or JSON files to `POST /documents`. The response contains an
`upload_id`; include it in `upload_ids` when posting a natural-language
`question` to `POST /questions`. Answers are restricted to the uploaded
documents identified by those IDs. Original files are stored under `uploads/`
and their chunks use the existing persistent Chroma collection.

Or configure the pipeline in Python:

```python
from document_pipeline import DocumentPipeline

pipeline = DocumentPipeline(r"C:\path\to\your\documents")
chunks = pipeline.run(prompt="What are the key points?")
```

The classes can also be used individually:

```python
from document_pipeline import (
    ChromaVectorStore,
    DocumentChunker,
    DocumentAgent,
    DocumentLoader,
    DocumentRetriever,
)

loader = DocumentLoader(r".\resources")
documents = loader.load_documents()
chunker = DocumentChunker()
chunks = chunker.split_documents(documents)

store_manager = ChromaVectorStore()
vector_store = store_manager.create_vector_store(chunks)
store_manager.print_stored_documents(chunks)

retriever = DocumentRetriever(vector_store)
agent = DocumentAgent(retriever)
agent.run("What are the key points?")
```

Relative document and Chroma paths are resolved from the current working
directory. PDFs are loaded as one document per page; each TXT or JSON file is
one document.
