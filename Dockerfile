FROM python:3.11-slim

# ffmpeg does the video work; the Noto font carries Arabic-script glyphs so
# burned-in Farsi subtitles render (libass handles the right-to-left shaping).
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        fonts-dejavu-core \
        fonts-noto-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# No torch. Both models run through CTranslate2 (a faster-whisper dependency),
# which is faster on CPU and needs no deep-learning framework alongside it.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Models land here. Mount a Railway volume at /data and they survive restarts
# instead of being re-downloaded on every deploy.
ENV HF_HOME=/data/hf \
    XDG_CACHE_HOME=/data/cache \
    WHISPER_MODEL=small \
    TRANSLATE_MODEL=OpenNMT/nllb-200-distilled-1.3B-ct2-int8

EXPOSE 8080
CMD streamlit run app.py \
      --server.port ${PORT:-8080} \
      --server.address 0.0.0.0 \
      --server.maxUploadSize 500 \
      --server.headless true \
      --browser.gatherUsageStats false
