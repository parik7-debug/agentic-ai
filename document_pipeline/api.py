import os
import tempfile
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field, field_validator

from document_pipeline.agent import DocumentAgent
from document_pipeline.chunker import DocumentChunker
from document_pipeline.loader import DocumentLoader
from document_pipeline.retriever import DocumentRetriever
from document_pipeline.vector_store import ChromaVectorStore


UPLOAD_DIRECTORY = Path(os.getenv("UPLOAD_DIRECTORY", "uploads"))
ALLOWED_FILE_SUFFIXES = {".pdf", ".txt", ".json"}
MAX_FILES_PER_UPLOAD = 10
MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024
MAX_TOTAL_UPLOAD_SIZE_BYTES = 50 * 1024 * 1024

app = FastAPI(
    title="Document Q&A API",
    description="Upload PDF, TXT, or JSON documents and ask questions about them.",
    version="1.0.0",
)


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    upload_ids: list[str] = Field(min_length=1, max_length=MAX_FILES_PER_UPLOAD)

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, question: str) -> str:
        if not question.strip():
            raise ValueError("Question must not be blank.")
        return question.strip()


class UploadResponse(BaseModel):
    upload_id: str
    files: list[str]
    chunks_indexed: int


class QuestionResponse(BaseModel):
    answer: str


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/documents", response_model=UploadResponse, status_code=201)
def upload_documents(
    files: Annotated[list[UploadFile], File(description="PDF, TXT, or JSON files")],
) -> UploadResponse:
    if not files:
        raise HTTPException(status_code=400, detail="Upload at least one document.")
    if len(files) > MAX_FILES_PER_UPLOAD:
        raise HTTPException(
            status_code=413,
            detail=f"Upload no more than {MAX_FILES_PER_UPLOAD} files at a time.",
        )

    uploads: list[tuple[str, str, bytes]] = []
    total_size = 0
    for uploaded_file in files:
        original_name = (uploaded_file.filename or "").replace("\\", "/").rsplit(
            "/", maxsplit=1
        )[-1]
        suffix = Path(original_name).suffix.lower()
        if not original_name or suffix not in ALLOWED_FILE_SUFFIXES:
            raise HTTPException(
                status_code=415,
                detail="Only PDF, TXT, and JSON documents are supported.",
            )

        content = uploaded_file.file.read(MAX_FILE_SIZE_BYTES + 1)
        if not content:
            raise HTTPException(
                status_code=422, detail=f"{original_name} is empty."
            )
        if len(content) > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"{original_name} exceeds the 20 MiB per-file limit.",
            )
        total_size += len(content)
        if total_size > MAX_TOTAL_UPLOAD_SIZE_BYTES:
            raise HTTPException(
                status_code=413,
                detail="The total upload size must not exceed 50 MiB.",
            )
        uploads.append((original_name, suffix, content))

    upload_id = uuid.uuid4().hex
    upload_root = UPLOAD_DIRECTORY.expanduser().resolve()
    upload_root.mkdir(parents=True, exist_ok=True)
    final_directory = upload_root / upload_id

    with tempfile.TemporaryDirectory(dir=upload_root) as staging_path:
        staging_directory = Path(staging_path)
        final_sources: dict[Path, tuple[Path, str]] = {}
        for index, (original_name, suffix, content) in enumerate(uploads):
            stored_name = f"{index:02d}-{uuid.uuid4().hex}{suffix}"
            staging_file = staging_directory / stored_name
            staging_file.write_bytes(content)
            final_sources[staging_file.resolve()] = (
                final_directory / stored_name,
                original_name,
            )

        documents = DocumentLoader(staging_directory).load_documents()
        if not documents:
            raise HTTPException(
                status_code=422, detail="No readable documents were uploaded."
            )

        for document in documents:
            source = Path(document.metadata["source"]).resolve()
            final_source = final_sources.get(source)
            if final_source is None:
                raise RuntimeError(f"Unexpected document source returned: {source}")
            stored_source, original_name = final_source
            document.metadata.update(
                {
                    "source": str(stored_source),
                    "file_name": original_name,
                    "original_filename": original_name,
                    "upload_id": upload_id,
                }
            )

        chunks = DocumentChunker().split_documents(documents)
        if not chunks:
            raise HTTPException(
                status_code=422, detail="The uploaded documents contained no text."
            )

        vector_store = ChromaVectorStore()
        vector_store.create_vector_store(chunks)
        staging_directory.rename(final_directory)

    return UploadResponse(
        upload_id=upload_id,
        files=[original_name for original_name, _, _ in uploads],
        chunks_indexed=len(chunks),
    )


@app.post("/questions", response_model=QuestionResponse)
def ask_question(request: QuestionRequest) -> QuestionResponse:
    vector_store = ChromaVectorStore().open_vector_store()
    metadata_filter = {"upload_id": {"$in": request.upload_ids}}
    stored_documents = vector_store.get(where=metadata_filter, limit=1)
    if not stored_documents["ids"]:
        raise HTTPException(
            status_code=404,
            detail="No indexed documents were found for the supplied upload IDs.",
        )

    retriever = DocumentRetriever(
        vector_store,
        metadata_filter=metadata_filter,
    )
    answer = DocumentAgent(retriever).answer(request.question)
    return QuestionResponse(answer=answer)
