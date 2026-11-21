from pathlib import Path

from langchain_core.documents import Document

from document_pipeline.agent import DocumentAgent
from document_pipeline.chunker import DocumentChunker
from document_pipeline.config import DEFAULT_DOCUMENT_LOCATION, QUESTION
from document_pipeline.loader import DocumentLoader
from document_pipeline.retriever import DocumentRetriever
from document_pipeline.vector_store import ChromaVectorStore


class DocumentPipeline:
    """Coordinate loading, chunking, vector storage, and retrieval."""

    def __init__(self, location: str | Path = DEFAULT_DOCUMENT_LOCATION) -> None:
        self.document_loader = DocumentLoader(location)
        self.chunker = DocumentChunker()
        self.vector_store = ChromaVectorStore()
        self.retriever: DocumentRetriever | None = None
        self.agent: DocumentAgent | None = None
        self.chunks: list[Document] = []

    def run(
        self,
        prompt: str = QUESTION,
        search_query: str | None = None,
    ) -> list[Document]:
        """Run the pipeline, then answer a prompt using the retrieval agent."""
        documents = self.document_loader.load_documents()
        self.document_loader.print_report()
        self.chunks = self.chunker.split_documents(documents)
        self.chunker.print_chunks()

        chroma_store = self.vector_store.create_vector_store(self.chunks)
        self.vector_store.print_stored_documents(self.chunks)
        self.vector_store.print_embeddings()

        self.retriever = DocumentRetriever(chroma_store)
        if search_query is not None:
            self.retriever.print_vector_store_results(search_query)
        self.agent = DocumentAgent(self.retriever)
        self.agent.run(prompt)
        return self.chunks
