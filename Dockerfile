# ParseFlow server image: Python + Tesseract OCR (+ LibreOffice for Word/PowerPoint/Excel layouts and DOC/PPT/XLS).
# Data (uploads, results, accounts DB) lives in /data. Build without LibreOffice (smaller image; DOCX/PPTX/XLSX/CSV
# are then rendered by ParseFlow itself): docker build --build-arg WITH_LIBREOFFICE=0 .
FROM python:3.12-slim
ARG WITH_LIBREOFFICE=0
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng fonts-dejavu-core \
    && if [ "$WITH_LIBREOFFICE" = "1" ]; then apt-get install -y --no-install-recommends libreoffice-writer-nogui \
       libreoffice-calc-nogui libreoffice-impress-nogui fonts-crosextra-carlito fonts-crosextra-caladea; fi \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
COPY requirements-postgres.txt .
RUN pip install --no-cache-dir -r requirements.txt -r requirements-postgres.txt
COPY backend backend
COPY frontend frontend
COPY .env.example .env.example
RUN mkdir -p /data && chmod 777 /data
ENV PARSEFLOW_DATA_DIR=/data PYTHONUNBUFFERED=1
EXPOSE 8000
WORKDIR /app/backend
# Supports both Render dynamic $PORT and default port 8000
CMD ["sh", "-c", "exec python -m uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips '*'"]
