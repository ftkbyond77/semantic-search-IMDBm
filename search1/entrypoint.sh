#!/bin/bash
set -e

# Log file for entrypoint actions
LOG_FILE="/app/logs/entrypoint.log"

# Ensure log directory exists
mkdir -p /app/logs
touch "$LOG_FILE"

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "Starting entrypoint script..."

# Wait for Neo4j to be available
log "Waiting for Neo4j to be available on neo4j:7687..."
timeout 60 bash -c 'while ! nc -z neo4j 7687; do sleep 1; done' || {
    log "ERROR: Neo4j failed to start within 60 seconds. Check Neo4j container logs."
    exit 1
}
log "Neo4j is up!"

# Wait for Redis to be available
log "Waiting for Redis to be available on redis:6379..."
timeout 60 bash -c 'while ! nc -z redis 6379; do sleep 1; done' || {
    log "ERROR: Redis failed to start within 60 seconds. Check Redis container logs."
    exit 1
}
log "Redis is up!"

# Run migrations
log "Running Django migrations..."
if ! python manage.py migrate > /app/logs/migration.log 2>&1; then
    log "ERROR: Django migrations failed. Check /app/logs/migration.log for details:"
    cat /app/logs/migration.log >> "$LOG_FILE"
    exit 1
fi
log "Django migrations completed successfully."

# Collect static files
log "Collecting static files..."
if ! python manage.py collectstatic --noinput > /app/logs/collectstatic.log 2>&1; then
    log "ERROR: Static file collection failed. Check /app/logs/collectstatic.log for details:"
    cat /app/logs/collectstatic.log >> "$LOG_FILE"
    exit 1
fi
log "Static files collected successfully."

# Optionally start Jupyter notebook (commented out as it may not be needed)
# log "Starting Jupyter notebook on port 8888..."
# jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' > /app/logs/jupyter.log 2>&1 &
# if [ $? -eq 0 ]; then
#     log "Jupyter notebook started successfully."
# else
#     log "ERROR: Failed to start Jupyter notebook. Check /app/logs/jupyter.log for details."
#     cat /app/logs/jupyter.log >> "$LOG_FILE"
# fi

# Start Django server
log "Starting Django development server on 0.0.0.0:8000..."
exec python manage.py runserver 0.0.0.0:8000