from django.shortcuts import render
from .models import Movie
from django.db.models import Q
from sentence_transformers import SentenceTransformer, util
import numpy as np
import torch

model = SentenceTransformer('all-MiniLM-L6-v2')

def home(request):
    search_type = request.GET.get('search_type', 'keyword')
    query = request.GET.get('q', '').strip()
    all_movies = Movie.objects.all()
    show_similarity = False

    if query:
        if search_type == 'keyword':
            movies = all_movies.filter(
                Q(series_title__icontains=query) |
                Q(genre__icontains=query) |
                Q(director__icontains=query) |
                Q(star1__icontains=query) |
                Q(star2__icontains=query)
            )
        elif search_type == 'semantic':
            try:
                query_embedding = model.encode(query)
                query_embedding = torch.tensor(query_embedding, dtype=torch.float32)

                filtered_movies = all_movies.exclude(embedding__isnull=True)
                similarities = []

                for movie in filtered_movies:
                    try:
                        movie_embedding = np.array(movie.embedding, dtype=np.float32)
                        movie_embedding = torch.tensor(movie_embedding, dtype=torch.float32)
                        similarity = util.cos_sim(query_embedding, movie_embedding).item()
                        similarities.append((movie, similarity))
                    except Exception as e:
                        print(f"Error processing movie {movie.id}: {e}")
                        continue

                if similarities:
                    similarities.sort(key=lambda x: x[1], reverse=True)
                    top_movies = similarities[:50]
                    movies = [{'movie': movie, 'similarity': round(sim * 100, 2)} for movie, sim in top_movies]
                    show_similarity = True
                else:
                    movies = []
            except Exception as e:
                print(f"Semantic search error: {e}")
                movies = []
        else:
            movies = []
    else:
        # No query yet, just show full list (like keyword)
        movies = all_movies

    return render(request, 'movies/home.html', {
        'movies': movies,
        'query': query,
        'search_type': search_type,
        'show_similarity': show_similarity
    })
