import os
import sys
import django
from import_dataset import download_imdb_dataset
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

# Configure Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'imdb_project.settings')
django.setup()

from movies.models import Movie

def load_imdb_data(clear_existing=False):
    try:
        # Load pre-trained model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        # Download dataset
        csv_file = download_imdb_dataset()
        df = pd.read_csv(csv_file)
        # Clear existing data if specified
        if clear_existing:
            Movie.objects.all().delete()
            print("Cleared existing data.")
        # Track existing titles
        existing_titles = set(Movie.objects.values_list('series_title', flat=True))
        for i, row in df.iterrows():
            if row['Series_Title'] in existing_titles:
                continue
            try:
                released_year = None
                try:
                    released_year = int(row['Released_Year'])
                except (ValueError, TypeError):
                    print(f"Warning: Invalid 'Released_Year' value '{row['Released_Year']}' for '{row['Series_Title']}', skipping.")
                    continue
                # Compute embedding as float32
                embedding = model.encode(row['Overview'], convert_to_numpy=True, dtype=np.float32).tolist()
                Movie.objects.create(
                    series_title=row['Series_Title'],
                    released_year=released_year,
                    runtime=row['Runtime'],
                    genre=row['Genre'],
                    rating=row['IMDB_Rating'],
                    overview=row['Overview'],
                    director=row['Director'],
                    star1=row['Star1'],
                    star2=row['Star2'],
                    embedding=embedding
                )
            except Exception as e:
                print(f"Error processing row for '{row.get('Series_Title', 'Unknown')}': {e}")
                continue
        print("Data and embeddings loaded successfully!")
    except Exception as e:
        print(f"Error loading data: {e}")

if __name__ == "__main__":
    load_imdb_data(clear_existing=True)