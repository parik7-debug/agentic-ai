import json

from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    RecursiveJsonSplitter,
)

from document_pipeline.config import CHUNK_OVERLAP, CHUNK_SIZE


class DocumentChunker:
    """Split loaded documents into text and structural JSON chunks."""

    def __init__(
        self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks: list[Document] = []

    def split_documents(self, documents: list[Document]) -> list[Document]:
        """Split JSON structurally and PDF/TXT content by character length."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        json_splitter = RecursiveJsonSplitter(max_chunk_size=self.chunk_size)
        self.chunks = []

        for document in documents:
            if document.metadata.get("file_type") != "json":
                self.chunks.extend(text_splitter.split_documents([document]))
                continue

            json_content = json.loads(document.page_content)
            if isinstance(json_content, (dict, list)):
                json_chunks = json_splitter.split_text(
                    json_content, convert_lists=True, ensure_ascii=False
                )
            else:
                json_chunks = json_splitter.split_text(
                    {"root": json_content}, ensure_ascii=False
                )

            self.chunks.extend(
                Document(page_content=chunk, metadata=document.metadata.copy())
                for chunk in json_chunks
            )
        return self.chunks

    def print_chunks(self) -> None:
        """Print each chunk and its source metadata."""
        print(
            f"\nCreated {len(self.chunks)} chunk(s). "
            f"JSON max chunk size: {self.chunk_size}; PDF/TXT chunk size: "
            f"{self.chunk_size}, overlap: {self.chunk_overlap}."
        )
        for index, chunk in enumerate(self.chunks, start=1):
            print(f"\nChunk {index} metadata:")
            print(
                json.dumps(chunk.metadata, ensure_ascii=False, indent=2, default=str)
            )
            print("Chunk content:")
            print(chunk.page_content)
