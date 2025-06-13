from django.shortcuts import render, get_object_or_404, redirect
from django.http import JsonResponse
from django.contrib.admin.views.decorators import staff_member_required
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib import messages
from django.core import serializers
from .models import Movie, Rating, SearchHistory
from django.db.models import Q, Count
from sentence_transformers import SentenceTransformer, util
import numpy as np
import torch
from scipy.sparse.linalg import svds
import warnings
import json
from django.utils import timezone
from .utils import SemanticKnowledgeGraph  

# Suppress Hugging Face FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.file_download")

# Load the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize SKG
skg = SemanticKnowledgeGraph()

def get_client_ip(request):
    """Get the client's IP address"""
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip

def get_or_create_session_id(request):
    """Get or create a session ID for anonymous users"""
    if not request.session.session_key:
        request.session.create()
    return request.session.session_key

def save_search_history(request, query, search_type, results_count):
    """Save search history with enhanced tracking"""
    search_data = {
        'query': query,
        'search_type': search_type,
        'results_count': results_count,
        'ip_address': get_client_ip(request),
        'user_agent': request.META.get('HTTP_USER_AGENT', ''),
        'timestamp': timezone.now()
    }
    
    if request.user.is_authenticated:
        search_data['user'] = request.user
    else:
        search_data['session_id'] = get_or_create_session_id(request)
    
    SearchHistory.objects.create(**search_data)
    print(f"Search saved: {query} ({search_type}) - {results_count} results")

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

def auth_view(request):
    """Handle authentication page"""
    if request.user.is_authenticated:
        return redirect('movies:home')
    
    return render(request, 'movies/auth.html')

def login_view(request):
    """Handle user login"""
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        
        if username and password:
            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
                messages.success(request, f'Welcome back, {user.username}!')
                return redirect('movies:home')
            else:
                messages.error(request, 'Invalid username or password.')
        else:
            messages.error(request, 'Please fill in all fields.')
    
    return render(request, 'movies/auth.html')

def signup_view(request):
    """Handle user registration"""
    if request.method == 'POST':
        username = request.POST.get('username')
        email = request.POST.get('email')
        password1 = request.POST.get('password1')
        password2 = request.POST.get('password2')
        
        if not all([username, email, password1, password2]):
            messages.error(request, 'Please fill in all fields.')
            return render(request, 'movies/auth.html')
        
        if password1 != password2:
            messages.error(request, 'Passwords do not match.')
            return render(request, 'movies/auth.html')
        
        if len(password1) < 8:
            messages.error(request, 'Password must be at least 8 characters long.')
            return render(request, 'movies/auth.html')
        
        if User.objects.filter(username=username).exists():
            messages.error(request, 'Username already exists.')
            return render(request, 'movies/auth.html')
        
        if User.objects.filter(email=email).exists():
            messages.error(request, 'Email already registered.')
            return render(request, 'movies/auth.html')
        
        try:
            user = User.objects.create_user(
                username=username,
                email=email,
                password=password1
            )
            login(request, user)
            messages.success(request, f'Welcome to IMDB Movie Explorer, {user.username}!')
            return redirect('movies:home')
        except Exception as e:
            messages.error(request, 'An error occurred during registration. Please try again.')
            print(f"Registration error: {e}")
    
    return render(request, 'movies/auth.html')

def logout_view(request):
    """Handle user logout"""
    logout(request)
    messages.success(request, 'You have been logged out successfully.')
    return redirect('movies:auth')

def home(request):
    is_guest = request.GET.get('guest') == 'true'
    
    if not request.user.is_authenticated and not is_guest:
        return redirect('movies:auth')
    
    search_type = request.GET.get('search_type', 'keyword')
    query = request.GET.get('q', '').strip()
    all_movies = Movie.objects.all()
    show_similarity = False
    recommendations = []
    search_suggestions = []
    results_count = 0
    match_explanation = None
    
    if query:
        if search_type == 'keyword':
            movies = all_movies.filter(
                Q(series_title__icontains=query) |
                Q(genre__icontains=query) |
                Q(director__icontains=query) |
                Q(star1__icontains=query) |
                Q(star2__icontains=query)
            )
            results_count = movies.count()
        elif search_type == 'semantic':
            try:
                # Query expansion using SKG
                expanded_terms = skg.expand_query(query)
                print(f"Expanded query terms: {expanded_terms}")
                combined_query = f"{query} {' '.join(expanded_terms)}"
                
                query_embedding = model.encode(combined_query)
                query_embedding = torch.tensor(query_embedding, dtype=torch.float32)

                filtered_movies = all_movies.exclude(embedding__isnull=True)
                similarities = []

                # Calculate average ratings for boosting
                movie_ratings = {r.movie_id: r.rating for r in Rating.objects.all()}
                # Calculate search frequency for boosting
                search_freq = SearchHistory.objects.filter(search_type='semantic').values('query').annotate(
                    count=Count('query')
                )
                query_freq = {s['query'].lower(): s['count'] for s in search_freq}
                query_count = query_freq.get(query.lower(), 1)

                for movie in filtered_movies:
                    try:
                        movie_embedding = np.array(movie.embedding, dtype=np.float32)
                        movie_embedding = torch.tensor(movie_embedding, dtype=torch.float32)
                        similarity = util.cos_sim(query_embedding, movie_embedding).item()
                        
                        # Signals boosting
                        rating_boost = movie_ratings.get(movie.id, 3.0) / 5.0  # Normalize rating (1-5)
                        freq_boost = min(np.log1p(query_count), 2.0)  # Cap frequency boost
                        boosted_similarity = similarity * (0.7 + 0.2 * rating_boost + 0.1 * freq_boost)
                        
                        similarities.append((movie, boosted_similarity, similarity))
                    except Exception as e:
                        print(f"Error processing movie {movie.id}: {e}")
                        continue

                if similarities:
                    similarities.sort(key=lambda x: x[1], reverse=True)
                    top_movies = similarities[:50]
                    movies = [
                        {
                            'movie': movie,
                            'similarity': round(boosted_similarity * 100, 2),
                            'original_similarity': round(original_sim * 100, 2),
                            'rating_boost': round(rating_boost, 2),
                            'freq_boost': round(freq_boost, 2)
                        } for movie, boosted_similarity, original_sim in top_movies
                    ]
                    show_similarity = True
                    results_count = len(movies)
                    # Store match explanation for detail view
                    match_explanation = {
                        'query': query,
                        'expanded_terms': expanded_terms,
                        'boost_factors': [
                            {'movie_id': m['movie'].id, 'rating_boost': m['rating_boost'], 'freq_boost': m['freq_boost']}
                            for m in movies
                        ]
                    }
                else:
                    movies = []
                    results_count = 0
            except Exception as e:
                print(f"Semantic search error: {e}")
                movies = []
                results_count = 0
        else:
            movies = []
            results_count = 0
        
        save_search_history(request, query, search_type, results_count)
    else:
        movies = all_movies
        results_count = movies.count()

    if request.user.is_authenticated:
        users = list(set(Rating.objects.values_list('user_id', flat=True)))
        movies_with_ratings = list(set(Rating.objects.values_list('movie_id', flat=True)))
        print(f"Users found: {len(users)}")
        print(f"Movies rated: {len(movies_with_ratings)}")
        
        if users and movies_with_ratings:
            ratings_matrix = np.zeros((len(users), len(movies_with_ratings)))
            for rating in Rating.objects.all():
                user_idx = users.index(rating.user_id)
                movie_idx = movies_with_ratings.index(rating.movie_id)
                ratings_matrix[user_idx, movie_idx] = rating.rating
            print(f"Ratings matrix created: {ratings_matrix}")
            
            recommendations = get_recommendations(
                user_id=request.user.id,
                ratings_matrix=ratings_matrix,
                user_ids=users,
                movie_ids=movies_with_ratings,
                movies_queryset=all_movies,
                num_recommendations=10
            )
        
        search_suggestions = get_search_based_suggestions(
            user_id=request.user.id,
            movies_queryset=all_movies,
            num_suggestions=5
        )

    print(f"Recommendations to template: {[m.series_title for m in recommendations]}")
    print(f"Search suggestions to template: {[m.series_title for m in search_suggestions]}")
    return render(request, 'movies/home.html', {
        'movies': movies,
        'query': query,
        'search_type': search_type,
        'show_similarity': show_similarity,
        'user_authenticated': request.user.is_authenticated,
        'recommendations': recommendations,
        'search_suggestions': search_suggestions,
        'results_count': results_count,
        'is_staff': request.user.is_staff if request.user.is_authenticated else False,
        'is_guest': is_guest,
        'match_explanation': match_explanation,
    })

def movie_detail(request, movie_id):
    movie = get_object_or_404(Movie, id=movie_id)
    ratings = Rating.objects.filter(movie=movie).select_related('user')
    user_rating = None
    match_explanation = request.session.get('match_explanation', None)
    boost_factors = None
    
    if match_explanation:
        for factor in match_explanation.get('boost_factors', []):
            if factor['movie_id'] == movie_id:
                boost_factors = {
                    'rating_boost': factor['rating_boost'],
                    'freq_boost': factor['freq_boost']
                }
                break
    
    if request.user.is_authenticated:
        user_rating = Rating.objects.filter(movie=movie, user=request.user).first()
    
    return render(request, 'movies/detail.html', {
        'movie': movie,
        'ratings': ratings,
        'user_rating': user_rating,
        'user_authenticated': request.user.is_authenticated,
        'match_explanation': match_explanation,
        'boost_factors': boost_factors,
    })

@staff_member_required
def export_search_data(request):
    """Export search history data as JSON"""
    search_history = SearchHistory.objects.all().select_related('user')
    
    data = []
    for search in search_history:
        data.append({
            'id': search.id,
            'user': search.user.username if search.user else None,
            'session_id': search.session_id,
            'query': search.query,
            'search_type': search.search_type,
            'results_count': search.results_count,
            'ip_address': search.ip_address,
            'user_agent': search.user_agent,
            'timestamp': search.timestamp.isoformat(),
        })
    
    return JsonResponse({
        'total_searches': len(data),
        'searches': data
    }, json_dumps_params={'indent': 2})

@staff_member_required
def search_analytics(request):
    """View search analytics"""
    from django.db.models import Count
    
    total_searches = SearchHistory.objects.count()
    keyword_searches = SearchHistory.objects.filter(search_type='keyword').count()
    semantic_searches = SearchHistory.objects.filter(search_type='semantic').count()
    
    popular_queries = SearchHistory.objects.values('query').annotate(
        count=Count('query')
    ).order_by('-count')[:10]
    
    user_search_counts = SearchHistory.objects.filter(user__isnull=False).values(
        'user__username'
    ).annotate(
        count=Count('user')
    ).order_by('-count')[:10]
    
    context = {
        'total_searches': total_searches,
        'keyword_searches': keyword_searches,
        'semantic_searches': semantic_searches,
        'popular_queries': popular_queries,
        'user_search_counts': user_search_counts,
    }
    
    return render(request, 'movies/search_analytics.html', context)