from document_pipeline.config import QUESTION
from document_pipeline.pipeline import DocumentPipeline


if __name__ == "__main__":
    prompt = input(f"Ask a question about the documents [{QUESTION}]: ").strip()
    DocumentPipeline().run(prompt=prompt or QUESTION)
