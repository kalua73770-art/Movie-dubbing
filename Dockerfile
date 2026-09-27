FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY hindi_dubbing ./hindi_dubbing
COPY app.py ./

RUN pip install --no-cache-dir .

ENV PYTHONUNBUFFERED=1
ENV PORT=10000
EXPOSE 10000

CMD ["sh","-c","uvicorn app:app --host 0.0.0.0 --port $PORT"]
