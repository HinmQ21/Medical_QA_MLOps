FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    RETRIEVAL_DEVICE=cpu \
    KG_DATA_DIR=/mnt/artifacts/smoke/kg \
    HF_HOME=/mnt/artifacts/hf \
    SENTENCE_TRANSFORMERS_HOME=/mnt/artifacts/hf/sentence-transformers

WORKDIR /app

RUN addgroup --system app && adduser --system --ingroup app app

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir '.[runtime]'

USER app
EXPOSE 8001

CMD ["uvicorn", "medical_qa_platform.retrieval.service:create_retrieval_service", "--factory", "--host", "0.0.0.0", "--port", "8001"]
