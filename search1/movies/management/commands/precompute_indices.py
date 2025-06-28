import os
import logging
import pickle
from django.core.management.base import BaseCommand
from movies.models import Movie
from rank_bm25 import BM25Okapi
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from django.conf import settings

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Precompute search indices for movies'

    def handle(self, *args, **kwargs):
        logging.basicConfig(
            filename='/app/logs/precompute_indices.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        logger.info("Starting index precomputation")

        def clean_text(text):
            if not text or isinstance(text, float):
                return ""
            try:
                return text.encode('utf-8', errors='replace').decode('utf-8')
            except Exception as e:
                logger.warning(f"Text cleaning error: {e}")
                return text.encode('ascii', errors='replace').decode('ascii')

        try:
            movies = Movie.objects.all()
            if not movies.exists():
                logger.error("No movies found in the database")
                raise ValueError("No movies found in the database")

            documents = []
            for movie in movies:
                try:
                    text = f"{clean_text(movie.series_title)} {clean_text(movie.overview)} {clean_text(movie.genre)}"
                    documents.append({
                        'id': movie.id,
                        'movie': movie,
                        'text': text
                    })
                except Exception as e:
                    logger.warning(f"Skipping movie {movie.id} due to error: {e}")
                    continue

            if not documents:
                logger.error("No valid documents in corpus")
                raise ValueError("No valid documents in corpus")

            logger.info(f"Processing {len(documents)} movies for indexing")

            tokenized_corpus = [doc['text'].lower().split() for doc in documents]
            bm25 = BM25Okapi(tokenized_corpus)
            logger.info("BM25 index created")

            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            model = AutoModel.from_pretrained('bert-base-uncased')
            embeddings = []
            for doc in documents:
                try:
                    inputs = tokenizer(doc['text'], return_tensors='pt', padding=True, truncation=True, max_length=512)
                    with torch.no_grad():
                        outputs = model(**inputs)
                    embedding = outputs.last_hidden_state.mean(dim=1).numpy()
                    embeddings.append(embedding)
                except Exception as e:
                    logger.warning(f"Error computing embedding for movie {doc['id']}: {e}")
                    embeddings.append(np.zeros((1, 768)))

            embeddings = np.vstack(embeddings).astype('float32')
            dimension = embeddings.shape[1]
            faiss_index = faiss.IndexFlatL2(dimension)
            faiss_index.add(embeddings)
            logger.info("FAISS index created")

            os.makedirs(settings.SEARCH_INDEX_PATH, exist_ok=True)
            index_path = os.path.join(settings.SEARCH_INDEX_PATH, 'search_indices.pkl')
            faiss_index_path = os.path.join(settings.SEARCH_INDEX_PATH, 'faiss_index.bin')

            with open(index_path, 'wb') as f:
                pickle.dump({'bm25': bm25, 'documents': documents}, f)
            faiss.write_index(faiss_index, faiss_index_path)
            logger.info(f"Indices saved to {index_path} and {faiss_index_path}")
            logger.info("Indices precomputed successfully")
        except Exception as e:
            logger.error(f"Error during index precomputation: {e}")
            raise
