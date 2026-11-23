import json
import unittest
from typing import cast
from unittest.mock import Mock, patch

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from document_pipeline.agent import DocumentAgent
from document_pipeline.chunker import DocumentChunker
from document_pipeline.retriever import DocumentRetriever


class DocumentChunkerTests(unittest.TestCase):
    def test_text_chunks_respect_size_and_preserve_metadata(self) -> None:
        metadata = {"source": "notes.txt", "file_type": "txt"}
        document = Document(
            page_content="alpha beta gamma delta epsilon zeta eta theta",
            metadata=metadata,
        )

        chunks = DocumentChunker(chunk_size=18, chunk_overlap=3).split_documents(
            [document]
        )

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk.page_content) <= 18 for chunk in chunks))
        self.assertTrue(all(chunk.metadata == metadata for chunk in chunks))

    def test_json_chunks_remain_valid_json_and_preserve_metadata(self) -> None:
        metadata = {"source": "topics.json", "file_type": "json"}
        document = Document(
            page_content=json.dumps(
                {
                    "topics": [
                        {"name": f"topic-{index}", "description": "economics"}
                        for index in range(4)
                    ]
                }
            ),
            metadata=metadata,
        )

        chunks = DocumentChunker(chunk_size=40).split_documents([document])

        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            json.loads(chunk.page_content)
            self.assertEqual(chunk.metadata, metadata)


class DocumentRetrieverTests(unittest.TestCase):
    def test_retrieval_formats_source_metadata_and_content(self) -> None:
        document = Document(
            page_content="Opportunity cost is the next best alternative.",
            metadata={"source": "notes.txt", "upload_id": "upload-1"},
        )
        fake_retriever = Mock()
        fake_retriever.invoke.return_value = [document]
        fake_store = Mock()
        fake_store.as_retriever.return_value = fake_retriever
        metadata_filter = {"upload_id": {"$in": ["upload-1"]}}

        retriever = DocumentRetriever(
            cast(Chroma, fake_store),
            search_result_count=2,
            metadata_filter=metadata_filter,
        )

        result = retriever.retrieve_documents("opportunity cost")

        fake_store.as_retriever.assert_called_once_with(
            search_kwargs={"k": 2, "filter": metadata_filter}
        )
        self.assertIn("Source: notes.txt", result)
        self.assertIn("Opportunity cost is the next best alternative.", result)
        self.assertIn('"upload_id": "upload-1"', result)


class DocumentAgentTests(unittest.TestCase):
    def test_answer_returns_final_ai_message_text(self) -> None:
        retriever = cast(DocumentRetriever, Mock())
        retriever.vector_store = object()
        agent = DocumentAgent(retriever)
        fake_agent = Mock()
        fake_agent.invoke.return_value = {
            "messages": [AIMessage(content="The next best alternative.")]
        }

        with patch.object(agent, "create_agent", return_value=fake_agent):
            answer = agent.answer("What is opportunity cost?")

        self.assertEqual(answer, "The next best alternative.")
        fake_agent.invoke.assert_called_once()


if __name__ == "__main__":
    unittest.main()
