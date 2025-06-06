import os
import sys
from django.core import management
from django.conf import settings
from import_dataset import download_imdb_dataset
import pandas as pd

# Configure Django settings manually if not set
if not settings.configured:
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'imdb_project.settings')
    try:
        from django import setup
        setup()
    except ImportError as e:
        print(f"Error importing Django: {e}")
        sys.exit(1)

from movies.models import Movie

def load_imdb_data():
    try:
        # Download dataset and get CSV path
        csv_file = download_imdb_dataset()
        df = pd.read_csv(csv_file)
        # Clear existing data (optional, remove if you want to append)
        Movie.objects.all().delete()
        for _, row in df.iterrows():
            try:
                # Clean 'Released_Year': convert to int, default to None if invalid
                released_year = None
                try:
                    released_year = int(row['Released_Year'])
                except (ValueError, TypeError):
                    print(f"Warning: Invalid 'Released_Year' value '{row['Released_Year']}' for '{row['Series_Title']}', skipping.")
                    continue  # Skip this row if year is invalid

                Movie.objects.create(
                    series_title=row['Series_Title'],
                    released_year=released_year,
                    runtime=row['Runtime'],
                    genre=row['Genre'],
                    rating=row['IMDB_Rating'],
                    overview=row['Overview'],
                    director=row['Director'],
                    star1=row['Star1'],
                    star2=row['Star2']
                )
            except Exception as e:
                print(f"Error processing row for '{row.get('Series_Title', 'Unknown')}': {e}")
                continue
        print("Data loaded successfully!")
    except Exception as e:
        print(f"Error loading data: {e}")

if __name__ == "__main__":
    load_imdb_data()