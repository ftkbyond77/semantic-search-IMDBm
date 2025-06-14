FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt

COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

COPY . .

EXPOSE 8000 8888 7474 7687

ENTRYPOINT ["./entrypoint.sh"]