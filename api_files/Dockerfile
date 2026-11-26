FROM python:3.13-slim

# Compilers for llama-cpp-python (built from source for this CPU); libgomp for xgboost / llama.cpp
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch first (avoids ~2 GB of CUDA libraries), then the pinned requirements
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Code, forecast model and RAG index
COPY . .

# Bake the embedding model and LLM into the image, then never contact Hugging Face at run time
RUN python download_models.py
ENV HF_HUB_OFFLINE=1

EXPOSE 7860

# One worker (the LLM is large; the assistant loads in a background thread), threads for concurrent requests,
# long timeout because CPU answers can take a while
CMD ["gunicorn", "-w", "1", "--threads", "4", "-b", "0.0.0.0:7860", "--timeout", "600", "app:freight_api"]
