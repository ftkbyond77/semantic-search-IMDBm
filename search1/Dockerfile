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

# Copy your Django project
COPY . .

# Expose the port and run the server
EXPOSE 8000
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]