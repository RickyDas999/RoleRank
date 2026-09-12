# RoleRank API image. Serves GET /health, GET /jobs, and POST /recommend.
#
# The image installs the sentence-transformers/torch libraries (a declared
# runtime dependency for the embedding ranker) but does NOT pre-download the
# all-MiniLM-L6-v2 model weights at build time. Weights are fetched from
# Hugging Face lazily, on the first embedding request, and cached inside the
# container at /root/.cache/huggingface. That cache is lost when the
# container is removed; mount a volume to persist/reuse it across runs:
#
#   docker run --rm -p 8000:8000 \
#     -v rolerank-hf-cache:/root/.cache/huggingface \
#     rolerank:milestone-1
#
# TF-IDF requests and GET /health never trigger a download.

FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY config ./config
COPY data ./data

RUN pip install --no-cache-dir .

EXPOSE 8000

CMD ["uvicorn", "rolerank.api:app", "--host", "0.0.0.0", "--port", "8000"]
