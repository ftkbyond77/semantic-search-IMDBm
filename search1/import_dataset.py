# import_dataset.py
import kagglehub
import pandas as pd
import os

def download_imdb_dataset():
    path = kagglehub.dataset_download("harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows")
    print("Path to dataset files:", path)
    csv_file = os.path.join(path, 'imdb_top_1000.csv')
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found at: {csv_file}")
    return csv_file

if __name__ == "__main__":
    try:
        csv_file = download_imdb_dataset()
        print(f"Dataset ready at: {csv_file}")
    except Exception as e:
        print(f"Error downloading dataset: {e}")