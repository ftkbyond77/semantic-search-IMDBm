FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt

# Copy the entrypoint script and make it executable
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Copy your Django project
COPY . .

# Expose ports for Django (8000) and Jupyter (8888)
EXPOSE 8000 8888

# Use the entrypoint script to start services
ENTRYPOINT ["./entrypoint.sh"]