from django.shortcuts import render
from .models import Movie
from django.db.models import Q
from sentence_transformers import SentenceTransformer, util
import numpy as np
import torch

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
                # Load model for semantic search
                model = SentenceTransformer('all-MiniLM-L6-v2')
                query_embedding = model.encode(query, convert_to_tensor=True)
                
                # Get all movies with embeddings
                all_movies = movies.exclude(embedding__isnull=True)
                
                if all_movies.exists():
                    # Compute cosine similarity for each movie
                    similarities = []
                    for movie in all_movies:
                        try:
                            # Convert movie embedding to tensor with consistent dtype
                            if isinstance(movie.embedding, (list, np.ndarray)):
                                movie_embedding = torch.tensor(movie.embedding, dtype=query_embedding.dtype)
                            else:
                                movie_embedding = torch.tensor(movie.embedding, dtype=query_embedding.dtype)
                            
                            # Ensure both tensors have the same device
                            movie_embedding = movie_embedding.to(query_embedding.device)
                            
                            # Compute similarity
                            similarity = util.cos_sim(query_embedding, movie_embedding).item()
                            similarities.append((movie, similarity))
                            
                        except Exception as e:
                            # Skip movies with problematic embeddings
                            print(f"Skipping movie {movie.id} due to embedding error: {e}")
                            continue
                    
                    if similarities:
                        # Sort by similarity (descending)
                        similarities.sort(key=lambda x: x[1], reverse=True)
                        # Get top results (e.g., top 50)
                        top_movies = [movie for movie, _ in similarities[:50]]
                        movies = all_movies.filter(id__in=[m.id for m in top_movies])
                    else:
                        # No valid similarities found, return empty queryset
                        movies = Movie.objects.none()
                        
            except Exception as e:
                # If semantic search fails, fall back to empty results
                print(f"Semantic search error: {e}")
                movies = Movie.objects.none()
    
    return render(request, 'movies/home.html', {
        'movies': movies,
        'query': query,
        'search_type': search_type
    })