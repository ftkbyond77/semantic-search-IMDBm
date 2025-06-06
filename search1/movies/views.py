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
    movies = Movie.objects.all()
    
    if query:
        if search_type == 'keyword':
            movies = movies.filter(
                Q(series_title__icontains=query) |
                Q(genre__icontains=query) |
                Q(director__icontains=query) |
                Q(star1__icontains=query) |
                Q(star2__icontains=query)
            )
        elif search_type == 'semantic':
            try:
                # Encode query embedding as float32 tensor
                query_embedding = model.encode(query)
                query_embedding = torch.tensor(query_embedding, dtype=torch.float32)

                all_movies = movies.exclude(embedding__isnull=True)
                if all_movies.exists():
                    similarities = []
                    for movie in all_movies:
                        try:
                            # Convert movie embedding to float32 tensor
                            movie_embedding = np.array(movie.embedding, dtype=np.float32)
                            movie_embedding = torch.tensor(movie_embedding, dtype=torch.float32)

                            # Compute cosine similarity
                            similarity = util.cos_sim(query_embedding, movie_embedding).item()
                            similarities.append((movie, similarity))
                        except Exception as e:
                            print(f"Error processing movie {movie.id}: {e}")
                            continue

                    if similarities:
                        similarities.sort(key=lambda x: x[1], reverse=True)
                        top_movies = [movie for movie, _ in similarities[:50]]
                        movies = all_movies.filter(id__in=[m.id for m in top_movies])
                    else:
                        movies = Movie.objects.none()
                else:
                    movies = Movie.objects.none()
            except Exception as e:
                print(f"Semantic search error: {e}")
                movies = Movie.objects.none()

    return render(request, 'movies/home.html', {
        'movies': movies,
        'query': query,
        'search_type': search_type
    })
