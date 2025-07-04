#!/bin/bash
set -e

echo "Waiting for Neo4j to be available..."
while ! nc -z neo4j 7687; do
  sleep 1
done
echo "Neo4j is up!"

echo "Waiting for Redis to be available..."
while ! nc -z redis 6379; do
  sleep 1
done
echo "Redis is up!"

echo "Running migrations..."
if ! python manage.py migrate > migration.log 2>&1; then
  echo "Migration failed. Check migration.log for details:"
  cat migration.log
  exit 1
fi
echo "Migrations completed successfully"

echo "Starting Jupyter notebook..."
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' &

echo "Starting Django server..."
python manage.py runserver 0.0.0.0:8000