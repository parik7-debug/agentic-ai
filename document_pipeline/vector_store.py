import hashlib
import json
import re
import time
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai._common import GoogleGenerativeAIError

from document_pipeline.config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIRECTORY,
    EMBEDDING_MODEL,
)
from document_pipeline.config import get_gemini_api_key


MAX_EMBEDDING_ATTEMPTS = 3
DEFAULT_RETRY_DELAY_SECONDS = 5
MAX_RETRY_DELAY_SECONDS = 120


class ChromaVectorStore:
    """Create and manage the persistent Chroma vector store."""

    def __init__(
        self,
        collection_name: str = CHROMA_COLLECTION_NAME,
        persist_directory: str | Path = CHROMA_PERSIST_DIRECTORY,
    ) -> None:
        self.collection_name = collection_name
        directory = Path(persist_directory).expanduser()
        if not directory.is_absolute():
            directory = Path.cwd() / directory
        self.persist_directory = directory.resolve()
        self.vector_store: Chroma | None = None
        self.embeddings: list[list[float]] = []

    @staticmethod
    def _is_rate_limit_error(error: GoogleGenerativeAIError) -> bool:
        cause = error.__cause__
        return getattr(cause, "code", None) == 429 or getattr(
            cause, "status", None
        ) == "RESOURCE_EXHAUSTED"

    @staticmethod
    def _get_retry_delay_seconds(
        error: GoogleGenerativeAIError, attempt: int
    ) -> float | None:
        cause_details = getattr(error.__cause__, "details", None)
        if isinstance(cause_details, dict):
            error_details = cause_details.get("error")
            if isinstance(error_details, dict):
                for detail in error_details.get("details", []):
                    if not isinstance(detail, dict):
                        continue
                    retry_delay = detail.get("retryDelay")
                    if isinstance(retry_delay, str):
                        match = re.fullmatch(r"(\d+(?:\.\d+)?)s", retry_delay)
                        if match:
                            delay = float(match.group(1))
                            return (
                                delay
                                if delay <= MAX_RETRY_DELAY_SECONDS
                                else None
                            )

        backoff = DEFAULT_RETRY_DELAY_SECONDS * (2 ** (attempt - 1))
        return min(backoff, MAX_RETRY_DELAY_SECONDS)

    def _create_embedding_model(self) -> GoogleGenerativeAIEmbeddings:
        return GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            api_key=get_gemini_api_key(),
            task_type="RETRIEVAL_DOCUMENT",
        )

    def open_vector_store(self) -> Chroma:
        """Open the persistent collection for retrieval."""
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            persist_directory=str(self.persist_directory),
            embedding_function=self._create_embedding_model(),
        )
        return self.vector_store

    def create_vector_store(self, chunks: list[Document]) -> Chroma | None:
        """Embed chunks and persist them in this store's collection."""
        self.vector_store = None
        self.embeddings = []
        if not chunks:
            return None

        embedding_model = self._create_embedding_model()
        document_ids = [
            hashlib.sha256(
                "\0".join(
                    (
                        str(chunk.metadata.get("source", "")),
                        str(chunk.metadata.get("page", "")),
                        str(index),
                        chunk.page_content,
                    )
                ).encode("utf-8")
            ).hexdigest()
            for index, chunk in enumerate(chunks)
        ]
        for attempt in range(1, MAX_EMBEDDING_ATTEMPTS + 1):
            try:
                self.vector_store = Chroma.from_documents(
                    documents=chunks,
                    embedding=embedding_model,
                    ids=document_ids,
                    collection_name=self.collection_name,
                    persist_directory=str(self.persist_directory),
                )
                break
            except GoogleGenerativeAIError as error:
                if (
                    not self._is_rate_limit_error(error)
                    or attempt == MAX_EMBEDDING_ATTEMPTS
                ):
                    raise

                delay = self._get_retry_delay_seconds(error, attempt)
                if delay is None:
                    print(
                        "Gemini requested a retry delay longer than "
                        f"{MAX_RETRY_DELAY_SECONDS} seconds; not retrying automatically.",
                        flush=True,
                    )
                    raise
                print(
                    f"\nGemini embedding rate limit reached. Retrying in "
                    f"{delay:.1f} seconds "
                    f"(attempt {attempt + 1}/{MAX_EMBEDDING_ATTEMPTS}).",
                    flush=True,
                )
                time.sleep(delay)

        if self.vector_store is None:
            raise RuntimeError("Chroma vector store creation did not complete.")

        stored_embeddings = self.vector_store.get(include=["embeddings"])["embeddings"]
        if stored_embeddings is not None:
            self.embeddings = [
                [float(value) for value in embedding]
                for embedding in stored_embeddings
            ]
        print(
            f"\nStored {len(chunks)} chunk(s) in Chroma collection "
            f"'{self.collection_name}' at {self.persist_directory}."
        )
        return self.vector_store

    def print_embeddings(self) -> None:
        """Print each stored embedding vector."""
        print(f"\nCreated {len(self.embeddings)} embedding(s).")
        for index, embedding in enumerate(self.embeddings, start=1):
            print(f"\nEmbedding {index} ({len(embedding)} dimensions):")
            print(embedding)

    def print_stored_documents(self, chunks: list[Document]) -> None:
        """Verify and print documents retrieved from Chroma."""
        if self.vector_store is None:
            print("\nNo Chroma vector store is available to verify.")
            return

        stored = self.vector_store.get(include=["documents", "metadatas"])
        ids = stored["ids"]
        documents = stored["documents"]
        metadatas = stored["metadatas"]
        if documents is None or metadatas is None:
            raise RuntimeError("Chroma did not return documents and metadata.")
        if not (len(ids) == len(documents) == len(metadatas) == len(chunks)):
            raise RuntimeError(
                f"Chroma returned {len(ids)} record(s) for {len(chunks)} chunk(s)."
            )

        stored_records = sorted(
            (
                content,
                json.dumps(
                    metadata or {},
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                ),
            )
            for content, metadata in zip(documents, metadatas)
        )
        expected_records = sorted(
            (
                chunk.page_content,
                json.dumps(
                    chunk.metadata,
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                ),
            )
            for chunk in chunks
        )
        if stored_records != expected_records:
            raise RuntimeError(
                "Retrieved Chroma documents or metadata do not match the split chunks."
            )

        print(f"\nVerified {len(ids)} document(s) retrieved from Chroma:")
        for index, (document_id, content, metadata) in enumerate(
            zip(ids, documents, metadatas), start=1
        ):
            print(f"\nStored document {index} (ID: {document_id}):")
            print("Metadata:")
            print(
                json.dumps(
                    metadata or {}, ensure_ascii=False, indent=2, default=str
                )
            )
            print("Content:")
            print(content)
