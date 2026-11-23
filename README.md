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

### Architecture and request workflow

```text
POST /documents
  -> DocumentLoader (PDF/TXT/JSON extraction)
  -> DocumentChunker
  -> Gemini embedding model
  -> Chroma collection: embedding_collection

POST /questions (question + upload_ids)
  -> Chroma retriever (filtered to those upload IDs)
  -> DocumentAgent / retrieve_documents tool
  -> Gemini chat model
  -> JSON answer
```

The API is a FastAPI layer over the same loader, chunker, vector-store,
retriever, and agent classes used by the command-line pipeline. Chroma and
upload directories can be configured with `CHROMA_PERSIST_DIRECTORY` and
`UPLOAD_DIRECTORY`; both default to local folders in the current working
directory.

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
and their chunks use the Chroma collection.

### Run unit tests

From the project root, run:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The tests cover API upload/question behavior, document chunking, retrieval
formatting, and agent responses. They mock Gemini and vector-store operations,
so they do not require a live API key or consume Gemini quota.

### Cloud Run demo deployment

The repository includes a `Dockerfile` and `.dockerignore` for Cloud Run.
Deploying requires a Google Cloud project with billing enabled, the Google
Cloud CLI (`gcloud`), and a Gemini API key. Do not put the key in the image,
source code, or a checked-in `.env` file.

1. Authenticate and select the Google Cloud project:

   ```powershell
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com secretmanager.googleapis.com
   ```

2. In Google Cloud Secret Manager, create a secret named `gemini-api-key` and
   add the Gemini key as a secret version. Grant the Cloud Run service identity
   the **Secret Manager Secret Accessor** role for this secret. Use the actual
   version number in the deploy command below.

3. From the project root, deploy a single-instance demo:

   ```powershell
   gcloud run deploy document-qa-api --source . --region us-central1 --no-allow-unauthenticated --memory 2Gi --timeout 300 --max 1 --concurrency 1 --set-env-vars "UPLOAD_DIRECTORY=/tmp/uploads,CHROMA_PERSIST_DIRECTORY=/tmp/chroma_store" --set-secrets "GEMINI_API_KEY=gemini-api-key:2"
   ```

   Replace `us-central1` and secret version `2` with your region and the current
   enabled secret version if they differ.
   `--no-allow-unauthenticated` keeps the service private.
   To make the service public, use `--allow-unauthenticated` instead only after
   considering the access and quota limitations below.

4. Copy the service URL printed by `gcloud run deploy`, then verify
   `https://SERVICE_URL/health`. For a private service, authenticate the request
   with an identity that has Cloud Run Invoker access. To give an evaluator
   access to the private service, grant their Google account the Invoker role:

   ```powershell
   gcloud run services add-iam-policy-binding document-qa-api --project YOUR_PROJECT_ID --region us-central1 --member user:EVALUATOR_EMAIL --role roles/run.invoker
   ```

The deployed capstone demo is currently at
[document-qa-api](https://document-qa-api-zmkwmubrgq-uc.a.run.app). It is private;
`/health` and `/docs` require an identity with Cloud Run Invoker access.

#### Test the deployed API (PowerShell)

The project owner must first grant the evaluator's Google account the
`roles/run.invoker` role using the command above. The evaluator then runs these
commands from the project root. Uploading and asking a question call Gemini and
may consume API quota.

```powershell
gcloud auth login
$url = "https://document-qa-api-zmkwmubrgq-uc.a.run.app"
$token = gcloud auth print-identity-token

curl.exe -i -H "Authorization: Bearer $token" "$url/health"

$upload = curl.exe -sS -H "Authorization: Bearer $token" `
  -F "files=@resources\AP_Micro_Unit1_Quizlet.txt" `
  "$url/documents" | ConvertFrom-Json
$upload | Format-List

$body = @{
  question = "What is opportunity cost?"
  upload_ids = @($upload.upload_id)
} | ConvertTo-Json -Compress

Invoke-RestMethod -Uri "$url/questions" -Method Post `
  -Headers @{ Authorization = "Bearer $token" } `
  -ContentType "application/json" -Body $body
```

The health request should return `{"status":"ok"}`. The upload response
includes an `upload_id` and `chunks_indexed`; the question response contains
the generated answer. Run the upload and question requests close together:
Cloud Run's demo storage is temporary, so an instance restart can remove the
uploaded chunks. Never share the identity token or Gemini API key.

For a local container smoke test:

```powershell
docker build -t document-qa-api .
docker run --rm -p 8080:8080 --env-file .env -e UPLOAD_DIRECTORY=/tmp/uploads -e CHROMA_PERSIST_DIRECTORY=/tmp/chroma_store document-qa-api
```

Then open `http://127.0.0.1:8080/docs`.

### Limitations and deployment considerations

- **Cloud Run data is temporary:** this demo writes uploads and Chroma files to
  `/tmp`. Cloud Run's writable filesystem uses instance memory and is lost when
  an instance stops. A new instance will not have the previous instance's
  uploads or vectors. The deployment command limits the service to one instance
  so consecutive requests are more likely to use the same in-memory store; it
  does not make the data durable. Re-upload documents after a restart or
  scale-to-zero event.
- **Memory use can grow:** uploads and Chroma's local database consume instance
  memory. The per-request upload limit does not impose a total storage limit.
- **No application-level user authentication:** upload IDs filter retrieval but
  are not user accounts or authorization. Keep Cloud Run IAM authentication
  enabled for this demo. If public access is required, add API authentication,
  rate limits, and usage monitoring before sharing the URL; otherwise others
  can use the Gemini-backed endpoints and consume your quota.
- **External API dependency:** embedding and answer generation require a valid
  Gemini key, network connectivity, available quota, and may incur charges.
- **Supported content:** PDF, TXT, and JSON only. Scanned PDFs require OCR,
  which is not included.
- **Demo, not production:** there is no document deletion endpoint,
  multi-user isolation, durable backup, or production monitoring.

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
