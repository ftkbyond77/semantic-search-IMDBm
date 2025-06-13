# Django IMDB Search Project

This project is a Django-based web application that allows users to search the IMDB Top 1000 Movies and TV Shows dataset using both **keyword** and **semantic** search. Semantic search leverages sentence embeddings to find movies based on thematic similarity (e.g., "prison escape"), while keyword search matches exact terms in titles, genres, directors, or stars. Cosine similarity scores are displayed for semantic search results, indicating how closely each movie matches the query.

## Features
- **Keyword Search**: Search movies by title, genre, director, or stars using case-insensitive partial matches.
- **Semantic Search**: Search movies by themes or concepts (e.g., "best movies", "prison escape") using sentence embeddings from the `all-MiniLM-L6-v2` model.
- **Cosine Similarity Scores**: Displays similarity scores (0 to 1) for semantic search results, formatted to three decimal places.
- **Responsive UI**: Modern interface with a magnifying glass search button, color-changing switch button (green for keyword, pink for semantic), and a gradient background.
- **Docker Support**: Runs in a containerized environment for consistent deployment.
- **Dataset Integration**: Automatically downloads the IMDB dataset via `kagglehub` and populates the database with embeddings.

## Project Structure
```
search1/
│
├── cleaned_imdb_data.csv               # Pre-cleaned movie dataset (CSV)
├── db.sqlite3                          # SQLite DB (auto-generated after migrate)
├── docker-compose.yml                  # Docker Compose config (services: Django, Jupyter)
├── Dockerfile                          # Image build for Django app
├── entrypoint.sh                       # Entrypoint script to run Django + Jupyter
├── import_dataset.py                   # (optional) Script for raw data import
├── load_data.py                        # Embedding generator script (e.g., Sentence-BERT)
├── manage.py                           # Django CLI manager
├── processing.ipynb                    # Notebook for embedding/testing/EDA
├── py1.ipynb                           # Another optional notebook
├── requirements.txt                    # Python dependencies (Django, numpy, transformers, etc.)
│
├── .ipynb_checkpoints/                 # Auto-generated Jupyter backups
│   └── processing-checkpoint.ipynb
│
├── db/                                 # Optional: for DB mount (empty or mapped volume)
│
├── imdb_project/                       # Django project root
│   ├── __init__.py
│   ├── asgi.py
│   ├── settings.py                     # Django settings
│   ├── urls.py                         # Project-level URL routing
│   ├── wsgi.py
│   └── __pycache__/                    # Compiled Python files
│       ├── settings.cpython-310.pyc
│       └── ...
│
├── movies/                             # Django app: Movies
│   ├── __init__.py
│   ├── admin.py                        # Admin interface for Movie model
│   ├── apps.py                         # App config
│   ├── models.py                       # Movie model (with embedding/vector fields)
│   ├── tests.py
│   ├── urls.py                         # App-level URLs (e.g., /movies/)
│   ├── views.py                        # Search and display logic
│
│   ├── management/
│   │   └── commands/
│   │       ├── seed_data.py           # Custom manage.py command to import CSV to DB
│   │       └── __pycache__/           # Compiled Python
│
│   ├── migrations/                     # Auto-generated migration files
│   │   ├── 0001_initial.py
│   │   ├── 0002_*.py
│   │   ├── ...
│   │   ├── __init__.py
│   │   └── __pycache__/
│   │       ├── *.pyc
│
│   ├── templates/
│   │   └── movies/
│   │       ├── auth.html              # Login/signup or access control page
│   │       ├── detail.html            # Movie detail page
│   │       ├── home.html              # Homepage with search bar
│   │       └── search_analytics.html  # (Optional) page for showing analytics/stats
│
│   └── __pycache__/                   # Compiled Python cache
│       ├── admin.cpython-310.pyc
│       ├── models.cpython-310.pyc
│       └── ...
│
├── static/                             # Static assets (CSS, JS, images)
├── staticfiles/                        # Collected static files (from collectstatic)
└── __pycache__/                        # Top-level pycache
    ├── load_data.cpython-310.pyc
    └── ...

```

## Prerequisites
- **Docker**: Required for containerized deployment.
- **Python 3.10**: Used in the Docker image (no local Python setup needed if using Docker).
- **Kaggle Account**: For `kagglehub` dataset download (set up `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables).

## Setup Instructions
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/ftkbyond77/semantic-search-IMDBm.git
   cd search1
   ```

2. **Set Up Kaggle Credentials**:
   - Obtain your Kaggle API token from [Kaggle](https://www.kaggle.com/settings/account).
   - Create a `.env` file in the project root:
     ```env
     KAGGLE_USERNAME=your_kaggle_username
     KAGGLE_KEY=your_kaggle_api_key
     ```
   - Alternatively, set these as environment variables in your system.

3. **Create Dockerfile**:
   - Create `Dockerfile` in the project root:
     ```dockerfile
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
     ```

4. **Create docker-compose.yml**:
   - Create `docker-compose.yml` in the project root:
     ```yaml
      services:
        app:
          build:
            context: .
            dockerfile: Dockerfile
          ports:
            - "8000:8000"  # Django server
            - "8888:8888"  # Jupyter Notebook
          volumes:
            - .:/app  # Mount the current directory to /app in the container
          environment:
            - PYTHONUNBUFFERED=1  # Ensure logs are output in real-time
          tty: true  # Enable interactive terminal for debugging
          stdin_open: true  # Keep STDIN open for interactive use
     ```

5. **Install Dependencies**:
   - Create `requirements.txt` in the project root:
     ```
      django==5.1.1
      kagglehub==0.3.1
      pandas==2.2.2
      numpy==1.26.4
      torch==2.3.0  
      transformers==4.40.0
      sentence-transformers==3.0.0
      scipy==1.13.0
      ipykernel
      jupyter
     ```

6. **Build and Run Docker**:
   ```bash
   docker-compose up --build
   ```

7. **Apply Migrations**:
   - In a new terminal, access the Docker container:
     ```bash
     docker-compose exec web bash
     ```
   - Run migrations:
     ```bash
     python manage.py makemigrations
     python manage.py migrate
     ```

8. **Load Dataset**:
   - Populate the database with the IMDB dataset and embeddings:
     ```bash
     docker-compose exec app python load_data.py
     ```

9. **Access the Application**:
   - Open http://localhost:8000 in your browser.
   - Use the search bar for keyword (e.g., "drama") or semantic (e.g., "prison escape") searches.
   - Toggle between search types using the switch button (green for keyword, pink for semantic).

## Usage
- **Keyword Search**:
  - Enter terms like "The Shawshank" or "Nolan" to find movies by title, genre, director, or stars.
  - Results display in a table with columns for Title, Year, Runtime, Genre, Rating, Director, and Stars.
- **Semantic Search**:
  - Enter themes like "best movies" or "prison escape" to find conceptually similar movies.
  - Results include an additional "Similarity Score" column showing cosine similarity (0 to 1, higher is more similar).
- **UI Features**:
  - **Search Button**: A magnifying glass icon triggers searches.
  - **Switch Button**: Toggles between keyword (green) and semantic (pink) search modes.
  - **Responsive Design**: Adapts to mobile and desktop screens with a gradient background and modern styling.

## Key Files
- **import_dataset.py**: Downloads the IMDB dataset using `kagglehub`.
- **load_data.py**: Loads the dataset into the database and generates float32 embeddings using `sentence-transformers`.
- **movies/models.py**: Defines the `Movie` model with fields for title, year, runtime, genre, rating, overview, director, stars, and embedding (JSONField).
- **movies/views.py**: Handles search logic, computing cosine similarity for semantic search and passing scores to the template.
- **movies/templates/movies/home.html**: Renders the search interface and results table with dynamic similarity scores for semantic search.

## Troubleshooting
- **No Results**:
  - Ensure `load_data.py` has run successfully:
    ```bash
    python manage.py shell
    from movies.models import Movie
    print(Movie.objects.count())
    print(Movie.objects.filter(embedding__isnull=False).count())
    ```
  - Reload data if needed: `python load_data.py`
- **Session Data Corrupted**:
  - Add to `imdb_project/settings.py`:
    ```python
    SESSION_ENGINE = 'django.contrib.sessions.backends.file'
    SESSION_FILE_PATH = BASE_DIR / 'sessions'
    ```
  - Create directory: `mkdir sessions`
  - Clear sessions: `python manage.py clearsessions`
- **Embedding Issues**:
  - Verify embeddings are float32:
    ```bash
    python manage.py shell
    from movies.models import Movie
    import numpy as np
    movie = Movie.objects.filter(embedding__isnull=False).first()
    print(np.array(movie.embedding).dtype)
    ```
  - Reset database if needed:
    ```bash
    rm db.sqlite3
    python manage.py makemigrations
    python manage.py migrate
    python load_data.py
    ```
- **Docker Logs**:
  - Check logs: `docker-compose logs`
- **Kaggle Errors**:
  - Ensure `KAGGLE_USERNAME` and `KAGGLE_KEY` are set correctly.

## Development Notes
- **Django Version**: 5.1.1
- **Python Version**: 3.10 (Docker)
- **Database**: SQLite (db.sqlite3)
- **Embedding Model**: `all-MiniLM-L6-v2` from `sentence-transformers`
- **Dataset**: [IMDB Top 1000 Movies and TV Shows](https://www.kaggle.com/datasets/harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows)
- **Logging**: Configurable in `settings.py` for debugging (e.g., `debug.log`).

## Future Improvements
- Move inline CSS to a static file for better maintainability.
- Add pagination for large result sets.
- Implement caching for embeddings to improve semantic search performance.
- Support advanced filters (e.g., by year, rating, or genre).
- Use a production-ready database like PostgreSQL.

## License
Apache License.

## Contact
For issues or suggestions, open a GitHub issue or contact the project maintainer.
