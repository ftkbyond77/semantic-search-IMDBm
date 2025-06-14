#!/bin/bash

# Install netcat if not present
apt-get update && apt-get install -y netcat-openbsd && rm -rf /var/lib/apt/lists/*

# Download spaCy model synchronously
if ! python -c "import spacy; spacy.load('en_core_web_sm')" 2>/dev/null; then
  echo "Downloading spaCy model en_core_web_sm..."
  python -m spacy download en_core_web_sm
fi

# Wait for Neo4j to be available
until nc -z neo4j 7687; do
  echo "Waiting for Neo4j to be available..."
  sleep 2
done

# Wait for Redis to be available
until nc -z redis 6379; do
  echo "Waiting for Redis to be available..."
  sleep 2
done

# Run Django migrations
python manage.py migrate

# Start the Django development server in the background
python manage.py runserver 0.0.0.0:8000 &

# Start the Jupyter Notebook server
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' &

# Keep the container running and wait for background processes
wait