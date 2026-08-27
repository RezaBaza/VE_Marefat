FROM python:3.11-slim

# ffmpeg does the video work; the Noto font carries Arabic-script glyphs so
# burned-in Farsi subtitles render (libass handles the right-to-left shaping).
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        fonts-dejavu-core \
        fonts-noto-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch: ~200MB instead of the ~2.5GB CUDA build we have no GPU for
RUN pip install --no-cache-dir torch==2.5.1 \
        --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Models land here. Mount a Railway volume at /data and they survive restarts
# instead of being re-downloaded on every deploy.
ENV HF_HOME=/data/hf \
    XDG_CACHE_HOME=/data/cache \
    WHISPER_MODEL=small \
    TRANSLATE_MODEL=facebook/nllb-200-distilled-600M

EXPOSE 8080
CMD streamlit run app.py \
      --server.port ${PORT:-8080} \
      --server.address 0.0.0.0 \
      --server.maxUploadSize 500 \
      --server.headless true \
      --browser.gatherUsageStats false
