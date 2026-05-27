FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

# uv pour gérer les dépendances exactement comme en local
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install dependencies first (better Docker layer caching)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# Copy source code and install the project
COPY src ./src
RUN uv sync --frozen --no-dev

# Le modèle MLflow et la BDD DuckDB sont attendus en volume monté à l'exécution :
#   docker run -v $(pwd)/mlflow.db:/app/mlflow.db \
#              -v $(pwd)/mlruns:/app/mlruns \
#              -v $(pwd)/data:/app/data \
#              -p 8000:8000 immoprix-api

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "mlops.api:app", "--host", "0.0.0.0", "--port", "8000"]
