# Slim image that runs the TriageMCP server over stdio (the MCP default transport).
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install dependencies first for better layer caching.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

# Drop privileges.
RUN useradd --create-home --uid 10001 triage
USER triage

# The API key is provided at runtime: docker run -e ANTHROPIC_API_KEY=... triagemcp
# Default command starts the MCP server on stdio.
ENTRYPOINT ["triagemcp"]
CMD ["serve"]
