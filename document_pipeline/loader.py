import json
import warnings
from pathlib import Path
from typing import Iterator

from langchain_community.document_loaders import (
    DirectoryLoader,
    JSONLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_community.document_loaders.base import BaseLoader
from langchain_core.documents import Document

from document_pipeline.config import DEFAULT_DOCUMENT_LOCATION


class _StandardJsonLoader(BaseLoader):
    """Load one JSON file without the native jq dependency."""

    def __init__(self, file_path: str | Path) -> None:
        self.file_path = Path(file_path)

    def lazy_load(self) -> Iterator[Document]:
        with self.file_path.open("r", encoding="utf-8") as file:
            content = json.dumps(json.load(file), ensure_ascii=False, indent=2)
        yield Document(page_content=content, metadata={"source": str(self.file_path)})


class DocumentLoader:
    """Load PDF, JSON, and TXT files into LangChain documents."""

    def __init__(self, location: str | Path = DEFAULT_DOCUMENT_LOCATION) -> None:
        directory = Path(location).expanduser()
        if not directory.is_absolute():
            directory = Path.cwd() / directory
        self.directory = directory.resolve()
        self.documents: list[Document] = []

    def load_documents(self) -> list[Document]:
        """Load supported files from the configured directory."""
        if not self.directory.is_dir():
            raise NotADirectoryError(
                f"Document location is not a directory: {self.directory}"
            )

        loader_configs = [
            ("**/*.[pP][dD][fF]", PyPDFLoader, {}),
            (
                "**/*.[tT][xX][tT]",
                TextLoader,
                {"encoding": "utf-8", "autodetect_encoding": True},
            ),
            (
                "**/*.[jJ][sS][oO][nN]",
                JSONLoader,
                {"jq_schema": ".", "text_content": False},
            ),
        ]
        self.documents = []
        for pattern, loader_class, loader_kwargs in loader_configs:
            try:
                loaded = DirectoryLoader(
                    str(self.directory),
                    glob=pattern,
                    loader_cls=loader_class,
                    loader_kwargs=loader_kwargs,
                    recursive=True,
                ).load()
            except ImportError as error:
                if loader_class is not JSONLoader:
                    raise
                warnings.warn(
                    "LangChain JSONLoader is unavailable because its jq "
                    f"dependency could not be loaded ({error}); loading JSON "
                    "files with Python's json module instead.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                loaded = DirectoryLoader(
                    str(self.directory),
                    glob=pattern,
                    loader_cls=_StandardJsonLoader,
                    recursive=True,
                ).load()

            for document in loaded:
                source = Path(document.metadata["source"])
                document.metadata.update(
                    {
                        "source": str(source),
                        "file_name": source.name,
                        "file_type": source.suffix.lower().lstrip("."),
                    }
                )
            self.documents.extend(loaded)

        self.documents.sort(
            key=lambda document: (
                document.metadata["source"],
                document.metadata.get("page", 0),
            )
        )
        return self.documents

    def print_report(self) -> None:
        """Print the loaded-document count and each document's metadata."""
        print(f"Loaded {len(self.documents)} document(s).")
        for index, document in enumerate(self.documents, start=1):
            print(f"\nDocument {index} metadata:")
            print(
                json.dumps(
                    document.metadata, ensure_ascii=False, indent=2, default=str
                )
            )
