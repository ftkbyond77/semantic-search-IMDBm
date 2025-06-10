import os
import django
import pandas as pd
from sentence_transformers import SentenceTransformer

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'imdb_project.settings')
django.setup()

from movies.models import Movie

def load_imdb_data(clear_existing=False, csv_file='cleaned_imdb_data.csv'):
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        df = pd.read_csv(csv_file)
        if clear_existing:
            Movie.objects.all().delete()
            print("Cleared existing data.")
        existing_titles = set(Movie.objects.values_list('series_title', flat=True))
        for i, row in df.iterrows():
            if row['Series_Title'] in existing_titles:
                continue
            try:
                released_year = int(row['Released_Year']) if pd.notnull(row['Released_Year']) else None
                embedding = model.encode(row['Overview'], convert_to_numpy=True).tolist()
                Movie.objects.create(
                    series_title=row['Series_Title'],
                    released_year=released_year,
                    certificate=row['Certificate'],
                    runtime=row['Runtime'],
                    genre=row['Genre'],
                    rating=row['IMDB_Rating'],
                    overview=row['Overview'],
                    meta_score=row['Meta_score'],
                    director=row['Director'],
                    star1=row['Star1'],
                    star2=row['Star2'],
                    star3=row['Star3'],
                    star4=row['Star4'],
                    no_of_votes=row['No_of_Votes'],
                    gross=row['Gross'],
                    poster_link=row['Poster_Link'],
                    embedding=embedding
                )
            except Exception as e:
                print(f"Error processing row for '{row.get('Series_Title', 'Unknown')}': {e}")
                continue
        print("Data and embeddings loaded successfully!")
    except Exception as e:
        print(f"Error loading data: {e}")

if __name__ == "__main__":
    import sys
    clear = False
    if len(sys.argv) > 1:
        clear = sys.argv[1].lower() in ['true', '1', 'yes']
    load_imdb_data(clear_existing=clear)