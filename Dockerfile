# Stage 1: Install Python dependencies
FROM python:3.12-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt
# openhands-sdk 1.19.1 (pinned by openhands-ai) passes rollout_entrypoint to
# lmnr's observe(); that kwarg was removed in lmnr>=0.7.53, which crashes the
# worker at import. Old lmnr can't be resolved normally (it exact-pins otel
# versions that conflict with the rest of the tree), but the runtime combo of
# lmnr 0.7.52 over the newer otel packages is verified working in prod — so
# force it in with --no-deps after the main resolution.
RUN /opt/venv/bin/pip install --no-cache-dir --no-deps lmnr==0.7.52

# Stage 2: Runtime
FROM python:3.12-slim

# System deps for WeasyPrint + LaTeX PDF + Node.js for OpenHands MCP (npx)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpango-1.0-0 libpangocairo-1.0-0 libpangoft2-1.0-0 \
        libcairo2 libgdk-pixbuf-2.0-0 libffi-dev shared-mime-info \
        fonts-dejavu-core fonts-liberation \
        texlive-base texlive-latex-recommended texlive-fonts-recommended \
        texlive-latex-extra texlive-fonts-extra lmodern \
        curl \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy Python venv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Copy application code
COPY backend/ backend/
COPY start.sh .
RUN chmod +x start.sh

EXPOSE 8080

CMD ["./start.sh"]
