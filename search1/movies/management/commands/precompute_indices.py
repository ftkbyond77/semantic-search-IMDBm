import os
import pickle
import numpy as np
from django.core.management.base import BaseCommand
from django.conf import settings
from movies.models import Movie
from rank_bm25 import BM25Okapi
import faiss
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

class Command(BaseCommand):
    help = 'Precompute BM25 and FAISS indices for hybrid search'

    def handle(self, *args, **options):
        self.stdout.write("Precomputing search indices...")

        # Initialize tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        model = AutoModel.from_pretrained('bert-base-uncased')

        # Fetch all movies
        movies = Movie.objects.all()
        documents = []
        corpus = []

        self.stdout.write(f"Processing {movies.count()} movies...")
        for movie in tqdm(movies, desc="Movies"):
            text = f"{movie.series_title} {movie.overview} {movie.genre} {movie.director} {' '.join([s for s in [movie.star1, movie.star2, movie.star3, movie.star4] if s])}"
            if hasattr(movie, 'awards'):
                text += f" {movie.awards}"
            documents.append({
                'id': movie.id,
                'movie': movie,
                'text': text
            })
            corpus.append(text.split())

        # Build BM25 index
        self.stdout.write("Building BM25 index...")
        bm25 = BM25Okapi(corpus)

        # Build FAISS index
        self.stdout.write("Building FAISS index...")
        dimension = 768  # BERT embedding dimension
        faiss_index = faiss.IndexFlatL2(dimension)

        # Compute embeddings in batches
        batch_size = 32
        embeddings = []
        texts = [doc['text'] for doc in documents]
        for i in tqdm(range(0, len(texts), batch_size), desc="Batches"):
            batch_texts = texts[i:i + batch_size]
            inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
            outputs = model(**inputs)
            batch_embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
            embeddings.append(batch_embeddings)

        embeddings = np.vstack(embeddings)
        faiss_index.add(embeddings)

        # Save indices
        index_path = os.path.join(settings.SEARCH_INDEX_PATH, 'search_indices.pkl')
        faiss_index_path = os.path.join(settings.SEARCH_INDEX_PATH, 'faiss_index.bin')

        self.stdout.write(f"Saving indices to {index_path} and {faiss_index_path}...")
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        with open(index_path, 'wb') as f:
            pickle.dump({
                'bm25': bm25,
                'documents': documents
            }, f)
        faiss.write_index(faiss_index, faiss_index_path)

        self.stdout.write(self.style.SUCCESS("Indices precomputed successfully!"))