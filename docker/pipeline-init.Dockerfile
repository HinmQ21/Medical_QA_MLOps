FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

RUN addgroup --system app && adduser --system --ingroup app app && \
    mkdir -p /workspace/.dvc /workspace/artifacts && \
    chown -R app:app /workspace

COPY pyproject.toml README.md ./
COPY src ./src
COPY mlops ./mlops

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir '.[pipeline]'

USER app

CMD ["dvc", "--version"]
