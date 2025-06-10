#!/bin/bash

# Start the Django development server in the background
python manage.py runserver 0.0.0.0:8000 &

# Start the Jupyter Notebook server
# Allow root access (common in Docker), bind to all interfaces, and use a default token for simplicity
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' &

# Keep the container running and wait for background processes
wait