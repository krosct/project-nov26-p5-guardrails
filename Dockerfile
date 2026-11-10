FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.10 /uv /bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY data/taxonomia.md ./data/taxonomia.md
RUN uv sync --locked --no-dev

# Modelo gerado por scripts/treinar.py
COPY models ./models

EXPOSE 8000
CMD ["/app/.venv/bin/uvicorn", "project_nov26_p5_guardrails.api:app", "--host", "0.0.0.0", "--port", "8000"]
