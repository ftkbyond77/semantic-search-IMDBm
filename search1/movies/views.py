from django.shortcuts import render, get_object_or_404
from .models import Movie, Rating, SearchHistory
from django.db.models import Q, Count
from sentence_transformers import SentenceTransformer, util
import numpy as np
import torch
from scipy.sparse.linalg import svds
import warnings

# Suppress Hugging Face FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.file_download")

# Load the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

def matrix_factorization(ratings_matrix, user_ids, movie_ids, k=5):
    print(f"Ratings matrix shape: {ratings_matrix.shape}")
    print(f"Ratings matrix non-zero entries: {np.count_nonzero(ratings_matrix)}")
    try:
        k = min(k, ratings_matrix.shape[0] - 1, ratings_matrix.shape[1] - 1)
        if k < 1:
            print("Matrix too small for SVD, returning empty predictions")
            return np.zeros(ratings_matrix.shape), None, None, None
        U, sigma, Vt = svds(ratings_matrix, k=k)
        sigma = np.diag(sigma)
        predicted_ratings = np.dot(np.dot(U, sigma), Vt)
        print(f"Predicted ratings shape: {predicted_ratings.shape}")
        return predicted_ratings, U, sigma, Vt
    except Exception as e:
        print(f"SVD error: {e}")
        return np.zeros(ratings_matrix.shape), None, None, None

def get_recommendations(user_id, ratings_matrix, user_ids, movie_ids, movies_queryset, num_recommendations=10):
    print(f"Generating recommendations for user_id: {user_id}")
    user_index = user_ids.index(user_id) if user_id in user_ids else -1
    if user_index == -1:
        print("User not found in user_ids")
        return []
    
    predicted_ratings, _, _, _ = matrix_factorization(ratings_matrix, user_ids, movie_ids)
    
    user_predicted_ratings = predicted_ratings[user_index]
    print(f"Predicted ratings for user: {user_predicted_ratings}")
    
    movie_ratings = []
    for idx, movie_id in enumerate(movie_ids):
        movie = movies_queryset.filter(id=movie_id).first()
        if movie:
            if not Rating.objects.filter(user_id=user_id, movie_id=movie_id).exists():
                movie_ratings.append((movie, user_predicted_ratings[idx]))
    
    movie_ratings.sort(key=lambda x: x[1], reverse=True)
    recommendations = [movie for movie, rating in movie_ratings[:num_recommendations]]
    print(f"Top {num_recommendations} recommendations: {[m.series_title for m in recommendations]}")
    return recommendations

def get_search_based_suggestions(user_id, movies_queryset, num_suggestions=5):
    frequent_searches = SearchHistory.objects.filter(user_id=user_id).values('query').annotate(
        count=Count('query')).order_by('-count')[:3]
    print(f"Frequent searches for user {user_id}: {[s['query'] for s in frequent_searches]}")
    
    suggestions = []
    for search in frequent_searches:
        query = search['query']
        matches = movies_queryset.filter(
            Q(series_title__icontains=query) |
            Q(genre__icontains=query) |
            Q(director__icontains=query) |
            Q(star1__icontains=query) |
            Q(star2__icontains=query)
        ).exclude(
            id__in=Rating.objects.filter(user_id=user_id).values('movie_id')
        )[:num_suggestions]
        suggestions.extend(matches)
    
    seen = set()
    unique_suggestions = []
    for movie in suggestions:
        if movie.id not in seen:
            unique_suggestions.append(movie)
            seen.add(movie.id)
    unique_suggestions = unique_suggestions[:num_suggestions]
    print(f"Search-based suggestions: {[m.series_title for m in unique_suggestions]}")
    return unique_suggestions

def home(request):
    search_type = request.GET.get('search_type', 'keyword')
    query = request.GET.get('q', '').strip()
    all_movies = Movie.objects.all()
    show_similarity = False
    recommendations = []
    search_suggestions = []
    
    if request.user.is_authenticated:
        if query:
            SearchHistory.objects.create(
                user=request.user,
                query=query,
                search_type=search_type
            )
        
        users = list(set(Rating.objects.values_list('user_id', flat=True)))
        movies = list(set(Rating.objects.values_list('movie_id', flat=True)))
        print(f"Users found: {len(users)}")
        print(f"Movies rated: {len(movies)}")
        
        if users and movies:
            ratings_matrix = np.zeros((len(users), len(movies)))
            for rating in Rating.objects.all():
                user_idx = users.index(rating.user_id)
                movie_idx = movies.index(rating.movie_id)
                ratings_matrix[user_idx, movie_idx] = rating.rating
            print(f"Ratings matrix created: {ratings_matrix}")
            
            recommendations = get_recommendations(
                user_id=request.user.id,
                ratings_matrix=ratings_matrix,
                user_ids=users,
                movie_ids=movies,
                movies_queryset=all_movies,
                num_recommendations=10
            )
        
        search_suggestions = get_search_based_suggestions(
            user_id=request.user.id,
            movies_queryset=all_movies,
            num_suggestions=5
        )

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
        movies = all_movies

    print(f"Recommendations to template: {[m.series_title for m in recommendations]}")
    print(f"Search suggestions to template: {[m.series_title for m in search_suggestions]}")
    return render(request, 'movies/home.html', {
        'movies': movies,
        'query': query,
        'search_type': search_type,
        'show_similarity': show_similarity,
        'user_authenticated': request.user.is_authenticated,
        'recommendations': recommendations,
        'search_suggestions': search_suggestions
    })

def movie_detail(request, movie_id):
    movie = get_object_or_404(Movie, id=movie_id)
    ratings = Rating.objects.filter(movie=movie).select_related('user')
    user_rating = None
    if request.user.is_authenticated:
        user_rating = Rating.objects.filter(movie=movie, user=request.user).first()
    return render(request, 'movies/detail.html', {
        'movie': movie,
        'ratings': ratings,
        'user_rating': user_rating,
        'user_authenticated': request.user.is_authenticated,
    })