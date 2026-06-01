FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

RUN addgroup --system app && adduser --system --ingroup app app && \
    mkdir -p /workspace/.dvc /workspace/artifacts && \
    chown -R app:app /workspace

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir 'dvc>=3.50'

USER app

CMD ["dvc", "--version"]
