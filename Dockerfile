# AS-ALD Co-Scientist - reproducible environment (ADR-008).
# Tier-0 (surface fidelity gate, selectivity model, verdict, manuscript) runs in this
# CPU image. For Tier-1 foundation-MLIP reactivity use a CUDA base image on Colab/GPU
# and `pip install .[mlip]`.
FROM python:3.12-slim

# tectonic for Layer-4 PDF compilation (optional; .tex is emitted regardless).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl ca-certificates \
    && curl -fsSL https://drop-sh.fullyjustified.net | sh -s -- --dest /usr/local/bin \
    || echo "tectonic install skipped (manuscript will emit .tex only)" \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e ".[openai]"

# Deterministic offline smoke test of the full funnel.
ENV PYTHONUNBUFFERED=1
CMD ["bash", "-lc", "aicoscientist --idea 'passivate a-SiN, grow SiOx-on-a-SiO2 to 90% at 10 nm' --offline --run-id demo --auto select:1 && aicoscientist-validate --run-id demo --offline && aicoscientist-paper --run-id demo"]
