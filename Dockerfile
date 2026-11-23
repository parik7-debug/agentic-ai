FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CHROMA_PERSIST_DIRECTORY=/tmp/chroma_store \
    UPLOAD_DIRECTORY=/tmp/uploads

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN adduser --disabled-password --gecos "" --uid 10001 appuser
USER appuser

EXPOSE 8080

CMD ["sh", "-c", "exec uvicorn document_pipeline.api:app --host 0.0.0.0 --port ${PORT:-8080}"]
