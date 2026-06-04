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

# uv = fast installs; CPU-only torch (retrieval runs CPU) avoids the ~2.5GB CUDA
# torch that sentence-transformers would otherwise pull. This layer precedes
# COPY src, so source-only changes reuse it from the build cache.
RUN pip install --no-cache-dir uv && \
    uv pip install --system --no-cache torch --index-url https://download.pytorch.org/whl/cpu

COPY src ./src

# torch already satisfied (CPU); '.[runtime]' adds faiss-cpu + sentence-transformers + numpy + the package
RUN uv pip install --system --no-cache '.[runtime]'

USER app
EXPOSE 8001

CMD ["uvicorn", "medical_qa_platform.retrieval.service:create_retrieval_service", "--factory", "--host", "0.0.0.0", "--port", "8001"]
