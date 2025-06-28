#!/bin/bash
echo "Starting entrypoint script..."

# Wait for PostgreSQL with timeout (30 seconds)
timeout=30
count=0
until pg_isready -h db -p 5432 -U user || nc -z db 5432; do
  echo "Waiting for PostgreSQL to be available on db:5432..."
  sleep 2
  ((count+=2))
  if [ $count -ge $timeout ]; then
    echo "ERROR: PostgreSQL not available after $timeout seconds"
    exit 1
  fi
done
echo "PostgreSQL is up!"

# Wait for Neo4j with timeout (30 seconds)
count=0
until nc -z neo4j 7687; do
  echo "Waiting for Neo4j to be available on neo4j:7687..."
  sleep 2
  ((count+=2))
  if [ $count -ge $timeout ]; then
    echo "ERROR: Neo4j not available after $timeout seconds"
    exit 1
  fi
done
echo "Neo4j is up!"

# Wait for Redis with timeout (30 seconds)
count=0
until nc -z redis 6379; do
  echo "Waiting for Redis to be available on redis:6379..."
  sleep 2
  ((count+=2))
  if [ $count -ge $timeout ]; then
    echo "ERROR: Redis not available after $timeout seconds"
    exit 1
  fi
done
echo "Redis is up!"

# Run migrations with retries
echo "Running database migrations..."
for i in {1..3}; do
  python manage.py migrate --noinput > /app/logs/migrate.log 2>&1
  if [ $? -eq 0 ]; then
    echo "Migrations successful"
    break
  else
    echo "Migration attempt $i failed, retrying..."
    sleep 5
  fi
done

# Check if migrations failed
if [ $? -ne 0 ]; then
  echo "ERROR: Database migrations failed. Check /app/logs/migrate.log for details"
  cat /app/logs/migrate.log
  exit 1
fi

# Run data import and index precomputation
echo "Importing IMDb data..."
python manage.py import_imdb_data > /app/logs/import_imdb_data.log 2>&1
if [ $? -ne 0 ]; then
  echo "ERROR: Data import failed. Check /app/logs/import_imdb_data.log for details"
  exit 1
fi

echo "Precomputing search indices..."
python manage.py precompute_indices > /app/logs/precompute_indices.log 2>&1
if [ $? -ne 0 ]; then
  echo "ERROR: Index precomputation failed. Check /app/logs/precompute_indices.log for details"
  exit 1
fi

# Start Gunicorn
echo "Starting Gunicorn..."
exec gunicorn --bind 0.0.0.0:8000 imdb_project.wsgi:application
