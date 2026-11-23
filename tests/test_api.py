import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from document_pipeline import api


class DocumentApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(api.app)

    def test_upload_indexes_supported_documents(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(api, "UPLOAD_DIRECTORY", Path(directory)),
                patch.object(api.ChromaVectorStore, "create_vector_store"),
            ):
                response = self.client.post(
                    "/documents",
                    files={
                        "files": (
                            "notes.txt",
                            b"Opportunity cost is the value of the next best alternative.",
                        )
                    },
                )

            self.assertEqual(response.status_code, 201)
            body = response.json()
            self.assertEqual(body["files"], ["notes.txt"])
            self.assertGreater(body["chunks_indexed"], 0)
            self.assertTrue((Path(directory) / body["upload_id"]).is_dir())

    def test_upload_rejects_unsupported_file_types(self) -> None:
        response = self.client.post(
            "/documents",
            files={"files": ("image.png", b"not a supported document")},
        )

        self.assertEqual(response.status_code, 415)

    def test_upload_rejects_empty_file(self) -> None:
        response = self.client.post(
            "/documents",
            files={"files": ("empty.txt", b"")},
        )

        self.assertEqual(response.status_code, 422)

    def test_upload_rejects_file_over_size_limit(self) -> None:
        with patch.object(api, "MAX_FILE_SIZE_BYTES", 3):
            response = self.client.post(
                "/documents",
                files={"files": ("large.txt", b"four")},
            )

        self.assertEqual(response.status_code, 413)

    def test_question_is_scoped_to_the_supplied_upload_ids(self) -> None:
        vector_store = Mock()
        vector_store.get.return_value = {"ids": ["chunk-1"]}
        manager = Mock()
        manager.open_vector_store.return_value = vector_store
        answerer = Mock()
        answerer.answer.return_value = "The next best alternative."
        upload_filter = {"upload_id": {"$in": ["upload-123"]}}

        with (
            patch.object(api, "ChromaVectorStore", return_value=manager),
            patch.object(api, "DocumentRetriever") as retriever_class,
            patch.object(api, "DocumentAgent", return_value=answerer),
        ):
            response = self.client.post(
                "/questions",
                json={
                    "question": "What is opportunity cost?",
                    "upload_ids": ["upload-123"],
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"answer": "The next best alternative."})
        vector_store.get.assert_called_once_with(where=upload_filter, limit=1)
        retriever_class.assert_called_once_with(
            vector_store, metadata_filter=upload_filter
        )
        answerer.answer.assert_called_once_with("What is opportunity cost?")

    def test_question_rejects_unknown_upload_ids(self) -> None:
        vector_store = Mock()
        vector_store.get.return_value = {"ids": []}
        manager = Mock()
        manager.open_vector_store.return_value = vector_store

        with (
            patch.object(api, "ChromaVectorStore", return_value=manager),
            patch.object(api, "DocumentRetriever") as retriever_class,
        ):
            response = self.client.post(
                "/questions",
                json={
                    "question": "What is opportunity cost?",
                    "upload_ids": ["unknown-upload"],
                },
            )

        self.assertEqual(response.status_code, 404)
        retriever_class.assert_not_called()


if __name__ == "__main__":
    unittest.main()
