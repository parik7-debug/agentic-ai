import os

from dotenv import load_dotenv


DEFAULT_DOCUMENT_LOCATION = "resources"
CHUNK_SIZE = 200
CHUNK_OVERLAP = 30
EMBEDDING_MODEL = "models/gemini-embedding-001"
CHROMA_COLLECTION_NAME = "embedding_collection"
CHROMA_PERSIST_DIRECTORY = "chroma_store"
DEFAULT_SEARCH_RESULT_COUNT = 4
QUESTION = "summarise the opportunity cost"
GENERATION_MODEL = "gemini-3.5-flash-lite"
REQUEST_TIMEOUT_SECONDS = 180


def get_gemini_api_key() -> str:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Add your Google Gemini API key to the .env file."
        )
    return api_key
