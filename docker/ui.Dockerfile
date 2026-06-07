FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN addgroup --system app && adduser --system --ingroup app app

COPY pyproject.toml README.md ./
COPY src ./src
COPY app ./app

RUN pip install --no-cache-dir uv && \
    uv pip install --system --no-cache '.[demo]'

USER app
EXPOSE 8501

CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port", "8501", "--server.address", "0.0.0.0"]
