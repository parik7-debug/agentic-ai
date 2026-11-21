import json
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from document_pipeline.config import DEFAULT_SEARCH_RESULT_COUNT


class DocumentRetriever:
    """Retrieve relevant documents from a Chroma vector store."""

    def __init__(
        self,
        vector_store: Chroma | None,
        search_result_count: int = DEFAULT_SEARCH_RESULT_COUNT,
        metadata_filter: dict[str, Any] | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.search_result_count = search_result_count
        self.metadata_filter = metadata_filter
        search_kwargs: dict[str, Any] = {"k": search_result_count}
        if metadata_filter is not None:
            search_kwargs["filter"] = metadata_filter
        self.retriever: BaseRetriever | None = (
            vector_store.as_retriever(search_kwargs=search_kwargs)
            if vector_store is not None
            else None
        )

    def retrieve_documents(self, query: str) -> str:
        """Return matching document chunks and metadata for an agent query."""
        if self.retriever is None:
            raise RuntimeError("No Chroma vector store is available to retrieve from.")
        if not query.strip():
            raise ValueError("Retrieval query must not be empty.")

        documents = self.retriever.invoke(query)
        if not documents:
            return "No relevant documents were found."

        return "\n\n".join(
            "\n".join(
                (
                    f"Source: {document.metadata.get('source', 'unknown')}",
                    "Metadata: "
                    + json.dumps(
                        document.metadata,
                        ensure_ascii=False,
                        sort_keys=True,
                        default=str,
                    ),
                    f"Content: {document.page_content}",
                )
            )
            for document in documents
        )

    def print_vector_store_results(
        self, query: str, k: int | None = None
    ) -> list[tuple[Document, float]]:
        """Search Chroma and print matching chunks, metadata, and distance scores."""
        if self.vector_store is None:
            print("\nNo Chroma vector store is available to search.")
            return []
        if not query.strip():
            raise ValueError("Search query must not be empty.")
        result_count = self.search_result_count if k is None else k
        if result_count < 1:
            raise ValueError("Search result count must be at least 1.")

        results = self.vector_store.similarity_search_with_score(
            query, k=result_count
        )
        print(f"\nChroma search results for: {query!r}")
        if not results:
            print("No matching documents found.")
            return results

        for index, (document, distance) in enumerate(results, start=1):
            print(f"\nResult {index} (distance: {distance:.6f}):")
            print("Metadata:")
            print(
                json.dumps(
                    document.metadata, ensure_ascii=False, indent=2, default=str
                )
            )
            print("Content:")
            print(document.page_content)
        return results
