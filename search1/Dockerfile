FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (postgresql-client and netcat-traditional)
RUN apt-get update && apt-get install -y \
    postgresql-client \
    netcat-traditional \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories with appropriate permissions
RUN mkdir -p /app/logs /app/static /app/staticfiles /app/search_indices && \
    chmod -R 755 /app/logs /app/static /app/staticfiles /app/search_indices

# Make entrypoint executable
RUN chmod +x /app/entrypoint.sh

# Expose port
EXPOSE 8000

# Set entrypoint and default command
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "imdb_project.wsgi:application"]
