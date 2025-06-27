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

# Wait for PostgreSQL to be available
log "Waiting for PostgreSQL to be available on db:5432..."
timeout 60 bash -c 'while ! nc -z db 5432; do sleep 1; done' || {
    log "ERROR: PostgreSQL failed to start within 60 seconds. Check db container logs."
    exit 1
}
log "PostgreSQL is up!"

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
log "Running database migrations..."
if ! python manage.py migrate > /app/logs/migrate.log 2>&1; then
    log "ERROR: Database migrations failed. Check /app/logs/migrate.log for details:"
    cat /app/logs/migrate.log >> "$LOG_FILE"
    exit 1
fi
log "Database migrations completed."

# Check if movies table is empty and import data if needed
log "Checking if movies_movie table is empty..."
MOVIE_COUNT=$(python manage.py shell -c "from movies.models import Movie; print(Movie.objects.count())")
if [ "$MOVIE_COUNT" -eq 0 ]; then
    log "Movies table is empty. Running data import commands..."

    # Import IMDb data
    log "Importing IMDb data..."
    if ! python manage.py import_imdb_data > /app/logs/import_imdb_data.log 2>&1; then
        log "ERROR: IMDb data import failed. Check /app/logs/import_imdb_data.log for details:"
        cat /app/logs/import_imdb_data.log >> "$LOG_FILE"
        exit 1
    fi
    log "IMDb data imported successfully."

    # Seed test user and ratings
    log "Seeding test user and ratings..."
    if ! python manage.py seed_data > /app/logs/seed_data.log 2>&1; then
        log "ERROR: Seed data failed. Check /app/logs/seed_data.log for details:"
        cat /app/logs/seed_data.log >> "$LOG_FILE"
        exit 1
    fi
    log "Test user and ratings seeded successfully."

    # Precompute search indices
    log "Precomputing BM25 and FAISS indices..."
    if ! python manage.py precompute_indices > /app/logs/precompute_indices.log 2>&1; then
        log "ERROR: Precompute indices failed. Check /app/logs/precompute_indices.log for details:"
        cat /app/logs/precompute_indices.log >> "$LOG_FILE"
        exit 1
    fi
    log "Search indices precomputed successfully."

    # Import data into Neo4j
    log "Importing data into Neo4j..."
    if ! python manage.py import_graph > /app/logs/import_graph.log 2>&1; then
        log "ERROR: Neo4j data import failed. Check /app/logs/import_graph.log for details:"
        cat /app/logs/import_graph.log >> "$LOG_FILE"
        exit 1
    fi
    log "Neo4j data imported successfully."
else
    log "Movies table contains $MOVIE_COUNT records. Skipping data import."
fi

# Collect static files
log "Collecting static files..."
if ! python manage.py collectstatic --noinput > /app/logs/collectstatic.log 2>&1; then
    log "ERROR: Static file collection failed. Check /app/logs/collectstatic.log for details:"
    cat /app/logs/collectstatic.log >> "$LOG_FILE"
    exit 1
fi
log "Static files collected successfully."

# Start Django server
log "Starting Django development server on 0.0.0.0:8000..."
exec python manage.py runserver 0.0.0.0:8000