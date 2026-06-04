FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

RUN addgroup --system app && adduser --system --ingroup app app && \
    mkdir -p /workspace/.dvc /workspace/artifacts && \
    chown -R app:app /workspace

RUN pip install --no-cache-dir uv && \
    uv pip install --system --no-cache 'dvc[gs]>=3.50'

USER app

CMD ["dvc", "--version"]
