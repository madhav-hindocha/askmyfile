# AskMyFile — container image for Hugging Face Spaces (or any Docker host).
#
# It installs the system programs the app needs (Tesseract for reading text
# out of scanned PDFs/images, plus libgomp1 which FAISS and PyTorch link
# against), installs the Python dependencies, and starts the Flask server.

FROM python:3.13-slim

# --- System dependencies -------------------------------------------------
# tesseract-ocr : lets the OCR feature read scanned PDFs and images
# libgomp1      : shared library required by faiss-cpu and torch
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# --- Non-root user (Hugging Face Spaces runs containers as uid 1000) ------
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PORT=7860 \
    HF_HOME=/home/user/.cache/huggingface \
    TRANSFORMERS_CACHE=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1

WORKDIR /app
RUN chown user:user /app

# --- Python dependencies (cached separately from app code) ----------------
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# --- Application code -----------------------------------------------------
COPY --chown=user . .
RUN chown -R user:user /app

USER user

EXPOSE 7860

# app.py reads PORT from the environment (7860 above) and, because PORT is
# set, runs in production mode (no debug, no reloader, host 0.0.0.0).
CMD ["python", "app.py"]
