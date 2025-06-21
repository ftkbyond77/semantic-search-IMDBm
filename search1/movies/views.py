from typing import List, Dict, Optional, Any, Tuple
from django.shortcuts import render, get_object_or_404, redirect
from django.http import JsonResponse, HttpResponse
from django.contrib.admin.views.decorators import staff_member_required
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib import messages
from django.core.paginator import Paginator
from django.db.models import Q, Count, Avg, QuerySet
from django.utils import timezone
from django import forms
from django.conf import settings
from django.views.decorators.http import require_GET
from .models import Movie, Rating, SearchHistory
from .utils import HybridSearch, SemanticKnowledgeGraph
import logging
import numpy as np
from scipy.sparse.linalg import svds
import redis
import pickle
import zlib
from neo4j import GraphDatabase

logger = logging.getLogger(__name__)

class RatingForm(forms.Form):
    rating = forms.ChoiceField(
        choices=[(i, f"{i} stars") for i in range(1, 6)],
        widget=forms.Select(attrs={'class': 'rating-select'}),
        label="Your Rating"
    )

def get_client_ip(request) -> str:
    """Get the client's IP address."""
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR', '')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR', '')
    return ip

def get_or_create_session_id(request) -> str:
    """Get or create a session ID for anonymous users."""
    if not request.session.session_key:
        request.session.save()  # Use save() instead of create()
    return request.session.session_key

def save_search_history(request, query: str, search_type: str, results_count: int) -> None:
    """Save search history with enhanced tracking."""
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

    try:
        SearchHistory.objects.create(**search_data)
        logger.info(f"Search saved: {query} ({search_type}) - {results_count} results")
    except Exception as e:
        logger.error(f"Error saving search history: {e}")

def matrix_factorization(ratings_matrix: np.ndarray, user_ids: List, movie_ids: List[int], k: int = 5) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Perform matrix factorization using SVD."""
    try:
        if ratings_matrix.size == 0:
            return np.zeros(ratings_matrix.shape), None, None, None
            
        if np.all(ratings_matrix == 0):
            return ratings_matrix, None, None, None
        U, sigma, Vt = svds(ratings_matrix, k=k)
        sigma = np.diag(sigma)
        predicted_ratings = np.dot(np.dot(U, sigma), Vt)
        logger.debug(f"Predicted ratings shape: {predicted_ratings.shape}")
        return predicted_ratings, U, sigma, Vt
    except Exception as e:
        logger.error(f"SVD error: {str(e)}")
        return np.zeros_like(ratings_matrix), None, None, None

def get_recommendations(request, movies_queryset: QuerySet, num_recommendations: int = 5) -> List[Movie]:
    """Generate personalized movie recommendations based on search history and ratings."""
    logger.info("Generating personalized recommendations")
    movies_queryset = movies_queryset.select_related().prefetch_related('user_ratings')
    user_id = request.user.id if request.user.is_authenticated else None
    session_id = get_or_create_session_id(request)
    skg = SemanticKnowledgeGraph()
    redis_client = redis.Redis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        db=0,
        decode_responses=True
    )
    redis_key = f"recs:{user_id or session_id}"
    cached = redis_client.get(redis_key)
    if cached:
        try:
            return pickle.loads(zlib.decompress(cached))
        except Exception as e:
            logger.error(f"Error deserializing cached recommendations: {e}")

    recommendations = []

    # Step 1: Get frequent search queries
    if user_id:
        frequent_searches = SearchHistory.objects.filter(user_id=user_id).values('query').annotate(
            count=Count('query')).order_by('-count')[:5]
    else:
        frequent_searches = SearchHistory.objects.filter(session_id=session_id).values('query').annotate(
            count=Count('query')).order_by('-count')[:5]

    logger.debug(f"Frequent searches: {[s['query'] for s in frequent_searches]}")

    # Step 2: Use search queries to find matching movies
    if frequent_searches:
        search_terms = [s['query'] for s in frequent_searches]
        q_objects = Q()
        for term in search_terms:
            q_objects |= (
                Q(series_title__icontains=term) |
                Q(genre__icontains=term) |
                Q(director__icontains=term) |
                Q(star1__icontains=term) |
                Q(star2__icontains=term) |
                Q(star3__icontains=term) |
                Q(star4__icontains=term) |
                Q(overview__icontains=term)
            )

        candidate_movies = movies_queryset.filter(q_objects).distinct()

        # Apply scores boost using user preferences
        movie_scores = []
        user_prefs = skg.user_preferences().get(user_id, {}) if user_id else {}
        for movie in candidate_movies:
            entities = skg._extract_movie_entities(movie)
            pref_boost = sum(user_prefs.get(entity, 0) for entity in entities) / (len(entities) or 1)
            popularity_boost = sum(skg.entity_popularity().get(entity, 0) for entity in entities) / 1000.0
            rating_boost = (movie.rating or 3.0) / 5.0
            score = 0.6 + 0.2 * rating_boost + 0.1 * popularity_boost + 0.1 * pref_boost
            movie_scores.append((movie, score))

        movie_scores.sort(key=lambda x: x[1], reverse=True)
        recommendations = [movie for movie, _ in movie_scores[:num_recommendations]]

    # Step 3: Fall back to matrix factorization if not enough recommendations
    if len(recommendations) < num_recommendations and user_id:
        users = list(set(Rating.objects.values_list('user_id', flat=True)))
        movies_with_ratings = list(set(Rating.objects.values_list('movie_id', flat=True)))
        if users and movies_with_ratings:
            ratings_matrix = np.zeros((len(users), len(movies_with_ratings)))
            for rating in Rating.objects.all():
                user_idx = users.index(rating.user_id)
                movie_idx = movies_with_ratings.index(rating.movie_id)
                ratings_matrix[user_idx, movie_idx] = rating.rating

            predicted_ratings, _, _, _ = matrix_factorization(
                ratings_matrix, users, movies_with_ratings
            )
            user_index = users.index(user_id) if user_id in users else -1

            if user_index != -1:
                user_predicted_ratings = predicted_ratings[user_index]
                movie_ratings = []
                for idx, movie_id in enumerate(movies_with_ratings):
                    movie = movies_queryset.filter(id=movie_id).first()
                    if movie and not Rating.objects.filter(user_id=user_id, movie_id=movie_id).exists():
                        entities = skg._extract_movie_entities(movie)
                        pref_boost = sum(user_prefs.get(entity, 0) for entity in entities) / (len(entities) or 1)
                        boosted_score = user_predicted_ratings[idx] * (1 + 0.2 * pref_boost)
                        movie_ratings.append((movie, boosted_score))

                movie_ratings.sort(key=lambda x: x[1], reverse=True)
                additional_recommendations = [
                    movie for movie, _ in movie_ratings[:num_recommendations - len(recommendations)]
                ]
                recommendations.extend(additional_recommendations)

    # Step 4: Fall back to popular movies
    if len(recommendations) < num_recommendations:
        popular_movies = movies_queryset.order_by(
            '-rating', '-no_of_votes'
        ).exclude(
            id__in=[movie.id for movie in recommendations]
        )[:num_recommendations - len(recommendations)]
        recommendations.extend(popular_movies)

    # Cache results with compression
    try:
        serialized = pickle.dumps(recommendations)
        compressed = zlib.compress(serialized)
        redis_client.setex(redis_key, 3600, compressed)
        logger.info(f"Cached recommendations: {[m.series_title for m in recommendations]}")
    except Exception as e:
        logger.error(f"Error caching recommendations: {e}")

    return recommendations

def get_search_based_suggestions(request, user_id: Optional[int], session_id: str, movies_queryset: QuerySet, num_suggestions: int = 5) -> List[Movie]:
    """Generate search-based movie suggestions based on history."""
    movies_queryset = movies_queryset.select_related()
    suggestions = []

    if user_id:
        frequent_searches = SearchHistory.objects.filter(user_id=user_id).values('query').annotate(
            count=Count('query')
        ).order_by('-count')[:3]
    else:
        frequent_searches = SearchHistory.objects.filter(session_id=session_id).values('query').annotate(
            count=Count('query')
        ).order_by('-count')[:3]

    logger.debug(f"Frequent searches for suggestions: {[s['query'] for s in frequent_searches]}")

    for search in frequent_searches:
        query = search['query']
        # Create base queryset with all the OR conditions
        q_objects = Q(
            Q(series_title__icontains=query) |
            Q(genre__icontains=query) |
            Q(director__icontains=query) |
            Q(star1__icontains=query) |
            Q(star2__icontains=query) |
            Q(star3__icontains=query) |
            Q(star4__icontains=query)
        )
        
        # Apply the filter and exclusion
        matches = movies_queryset.filter(q_objects)
        
        # Exclude movies already rated by the user if user_id exists
        if user_id:
            matches = matches.exclude(
                id__in=Rating.objects.filter(user_id=user_id).values_list('movie_id', flat=True)
            )
        
        # Get the specified number of suggestions
        matches = matches[:num_suggestions]
        suggestions.extend(matches)

    # Remove duplicates while preserving order
    seen = set()
    unique_suggestions = []
    for movie in suggestions:
        if movie.id not in seen:
            unique_suggestions.append(movie)
            seen.add(movie.id)
    
    # Return only the requested number of suggestions
    suggestions = unique_suggestions[:num_suggestions]

    logger.debug(f"Search suggestions: {[s.series_title for s in suggestions]}")
    return suggestions

def get_movie_stats() -> Dict[str, Any]:
    """Get general movie statistics."""
    try:
        return {
            'total_movies': Movie.objects.count(),  # Removed extra parenthesis
            'avg_rating': Movie.objects.aggregate(avg_rating=Avg('rating'))['avg_rating'] or 0.0,
            'genres': Movie.objects.values('genres').annotate(count=Count('genres')).order_by('-count')[:5],
            'recent_years': Movie.objects.filter(released_year__gte=timezone.now().year - 5).count(),
            'total_ratings': Rating.objects.count(),
        }
    except Exception as e:
        logger.error(f"Error getting movie stats: {str(e)}")
        return {}

def auth_view(request):
    """Handle authentication page."""
    if request.user.is_authenticated:
        return redirect('movies:home')

    return render(request, 'movies/auth.html')

def login_view(request):
    """Handle user login."""
    if request.method == 'POST':
        username = request.POST.get('username', '').strip()
        password = request.POST.get('password', '')

        if username and password:
            user = authenticate(request, username=username, password=password)
            if user:
                login(request, user)
                messages.success(request, f"Welcome back, {user.username}!")
                return redirect('movies:home')
            else:
                messages.error(request, 'Invalid username or password')
        else:
            messages.error(request, 'Please fill in all fields')

    return render(request, 'movies/auth.html')

def signup_view(request):
    """Handle user registration."""
    if request.method == 'POST':
        username = request.POST.get('username', '').strip()
        email = request.POST.get('email', '').strip()
        password1 = request.POST.get('password1', '').strip()
        password2 = request.POST.get('password2', '').strip()

        if not all([username, email, password1, password2]):
            messages.error(request, 'Please fill in all fields')
            return render(request, 'movies/auth.html')

        if password1 != password2:
            messages.error(request, 'Passwords do not match')
            return redirect('movies:auth.html')

        if len(password1) < 8:
            messages.error(request, 'Password must be at least 8 characters long')
            return render(request, 'movies/auth.html')

        if User.objects.filter(username=username).exists():
            messages.error(request, 'Username already exists')
            return redirect('movies:auth')

        if email and User.objects.filter(email=email).exists():
            messages.error(request, 'Email already registered')
            return redirect('movies:auth')

        try:
            user = User.objects.create_user(
                username=username,
                email=email,
                password=password1,
            )
            login(request, user)
            messages.success(request, f"Welcome to IMDb Movie Explorer, {user.username}!")
            return redirect('movies:home')
        except Exception as e:
            logger.error(f"Registration error: {str(e)}")
            messages.error(request, 'An error occurred during registration. Please try again')
            return render(request, 'movies/auth.html')

    return render(request, 'movies/auth.html')

def logout_view(request) -> HttpResponse:
    """Handle user logout request."""
    logout(request)
    messages.success(request, 'You have been logged out successfully')
    return redirect('movies:auth')

def home(request) -> HttpResponse:
    """Handle the main search page."""
    is_guest = request.GET.get('guest') == 'true'

    if not request.user.is_authenticated and not is_guest:
        return redirect('movies:auth')

    search_type = request.GET.get('search_type', 'keyword')
    query = request.GET.get('q', '').strip()
    sort_by = request.GET.get('sort', 'rating')
    page_number = int(request.GET.get('page', 1))

    all_movies = Movie.objects.select_related().prefetch_related('user_ratings')
    show_similarity = False
    recommendations = []
    search_suggestions = []
    results_count = 0
    match_explanation = None
    skg = SemanticKnowledgeGraph()
    trending_concepts = skg.get_trending_concepts(days=7, limit=5)
    movie_stats = get_movie_stats()
    movies = []

    try:
        if query:
            user_id = request.user.id if request.user.is_authenticated else None
            if search_type == 'keyword':
                query_terms = query.lower().split()
                q_objects = Q()
                for term in query_terms:
                    q_objects |= (
                        Q(series_title__icontains=term) |
                        Q(genre__icontains=term) |
                        Q(director__icontains=term) |
                        Q(star1__icontains=term) |
                        Q(star2__icontains=term) |
                        Q(star3__icontains=term) |
                        Q(star4__icontains=term) |
                        Q(overview__icontains=term)
                    )

                movies = all_movies.filter(q_objects).distinct()
                results_count = movies.count()

                if sort_by == 'rating':
                    movies = movies.order_by('-rating')
                elif sort_by == 'year':
                    movies = movies.order_by('-released_year')
                else:
                    movies = movies.order_by('series_title')

            elif search_type == 'semantic':
                searcher = HybridSearch()
                expanded_terms = skg.expand_query(query, user_id=user_id)
                combined_query = f"{query} {' '.join(expanded_terms)}"
                search_results = searcher.search(combined_query, user_id=user_id, top_k=50, rerank_k=10)

                movies = [
                    {
                        'movie': result['movie'],
                        'similarity': min(100, result['score'] * 100),
                        'explanations': skg.get_recommendation_explanation(query, expanded_terms, result['movie'])
                    }
                    for result in search_results
                ]
                show_similarity = True
                results_count = len(movies)

                match_explanation = {
                    'query': query,
                    'expanded_terms': expanded_terms,
                    'boost_factors': [
                        {
                            'movie_id': m['movie'].id,
                            'explanations': m['explanations']
                        }
                        for m in movies
                    ]
                }
                request.session['match_explanation'] = match_explanation

            save_search_history(request, query, search_type, results_count)
            if user_id:
                skg.update_from_user_interaction(user_id, None, 'search')
        else:
            movies = all_movies.order_by('-rating', '-no_of_votes')[:10]
            results_count = len(movies)

        paginator = Paginator(movies, 12)
        movies = paginator.get_page(page_number)

        user_id = request.user.id if request.user.is_authenticated else None
        session_id = get_or_create_session_id(request)
        recommendations = get_recommendations(request, all_movies, num_recommendations=5)
        search_suggestions = get_search_based_suggestions(request, user_id, session_id, all_movies, num_suggestions=5)

    except Exception as e:
        logger.error(f"Home page error: {str(e)}")
        messages.error(request, 'An error occurred while processing your request.')
        movies = []
        results_count = 0

    logger.info(f"Rendering home page: query='{query}', results_count={results_count}, search_type={search_type}")
    return render(request, 'movies/home.html', {
        'movies': movies,
        'query': query,
        'search_type': search_type,
        'sort_by': sort_by,
        'show_similarity': show_similarity,
        'user_authenticated': request.user.is_authenticated,
        'user': request.user,
        'recommendations': recommendations,
        'search_suggestions': search_suggestions,
        'results_count': results_count,
        'is_staff': request.user.is_staff if request.user.is_authenticated else False,
        'is_guest': is_guest,
        'trending_concepts': trending_concepts,
        'movie_stats': movie_stats,
    })

def movie_detail(request, movie_id: int) -> HttpResponse:
    """Handle movie detail page."""
    try:
        movie = get_object_or_404(Movie.objects.select_related(), id=movie_id)
        ratings = Rating.objects.filter(movie=movie).select_related('user')
        user_rating = None
        match_explanation = request.session.get('match_explanation', None)
        boost_factors = None
        rating_form = RatingForm()

        avg_rating = ratings.aggregate(avg=Avg('rating'))['avg'] or 0
        rating_distribution = {i: ratings.filter(rating=i).count() for i in range(1, 6)}

        related_movies = Movie.objects.select_related().filter(
            Q(genre__icontains=movie.genre.split(',')[0].strip()) |
            Q(director__icontains=movie.director)
        ).exclude(id=movie_id)[:6]

        if match_explanation:
            for factor in match_explanation.get('boost_factors', []):
                if factor['movie_id'] == movie_id:
                    boost_factors = {
                        'explanations': factor['explanations']
                    }
                    break

        if request.user.is_authenticated:
            user_rating = Rating.objects.filter(movie=movie, user=request.user).first()
            
            # Handle rating submission
            if request.method == 'POST':
                rating_form = RatingForm(request.POST)
                if rating_form.is_valid():
                    rating_value = int(rating_form.cleaned_data['rating'])
                    Rating.objects.update_or_create(
                        user=request.user,
                        movie=movie,
                        defaults={'rating': rating_value, 'timestamp': timezone.now()}
                    )
                    # Update knowledge graph after rating
                    skg = SemanticKnowledgeGraph()
                    skg.update_from_user(request.user.id, movie_id, 'rating')
                    messages.success(request, 'Rating updated successfully!')
                    return redirect('movies:movie_detail', movie_id=movie_id)

            # Track view in knowledge graph
            skg = SemanticKnowledgeGraph()
            skg.update_from_user(request.user.id, movie_id, 'view')

        return render(request, 'movies/detail.html', {
            'movie': movie,
            'ratings': ratings,
            'user_rating': user_rating,
            'user_authenticated': request.user.is_authenticated,
            'match_explanation': match_explanation,
            'boost_factors': boost_factors,
            'rating_form': rating_form,
            'avg_rating': round(avg_rating, 1),
            'rating_distribution': rating_distribution,
            'related_movies': related_movies,
            'total_ratings': ratings.count(),
        })
    except Exception as e:
        logger.error(f"Movie detail error for ID {movie_id}: {str(e)}")
        messages.error(request, 'Unable to load movie details.')
        return redirect('movies:home')

@staff_member_required
def export_search_data(request) -> HttpResponse:
    """Export search history data as JSON."""
    try:
        search_history = SearchHistory.objects.select_related('user').all()
        data = [
            {
                'id': search.id,
                'user': search.user.username if search.user else None,
                'session_id': search.session_id,
                'query': search.query,
                'search_type': search.search_type,
                'results_count': search.results_count,
                'ip_address': search.ip_address,
                'user_agent': search.user_agent,
                'timestamp': search.timestamp.isoformat(),
            }
            for search in search_history
        ]

        return JsonResponse({
            'total_searches': len(data),
            'searches': data
        }, json_dumps_params={'indent': 2})
    except Exception as e:
        logger.error(f"Export search data error: {str(e)}")
        return JsonResponse({'error': 'Failed to export search data'}, status=500)

@staff_member_required
def search_analytics(request) -> HttpResponse:
    """Render search analytics dashboard."""
    try:
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

        skg = SemanticKnowledgeGraph()
        trending_concepts = skg.get_trending_concepts(days=7, limit=10)

        context = {
            'total_searches': total_searches,
            'keyword_searches': keyword_searches,
            'semantic_searches': semantic_searches,
            'popular_queries': popular_queries,
            'user_search_counts': user_search_counts,
            'trending_concepts': trending_concepts,
        }

        return render(request, 'movies/search_analytics.html', context)
    except Exception as e:
        logger.error(f"Search analytics error: {str(e)}")
        messages.error(request, 'Unable to load analytics dashboard.')
        return redirect('movies:home')

def search_view(request) -> HttpResponse:
    """Handle search requests (alternative to home view)."""
    try:
        query = request.GET.get('q', '').strip()
        search_type = request.GET.get('search_type', 'keyword')
        sort_by = request.GET.get('sort', 'rating')
        page = request.GET.get('user_id')
        user_id = request.user_id if request.user.is_authenticated else None
        is_guest = not user_id or user_id.is_anonymous
        is_staff = user_id.is_staff if user_id else False

        movies = []
        recommendations = []
        search_suggestions = []
        show_similarity = search_type == 'semantic'
        if query and not is_guest:
            save_search_results(request, query, search_type, results_count=0)

        all_movies = Movie.objects.select_related().prefetch_related('user_ratings')
        results_count = movie_count = 0

        if query:
            if search_type == 'semantic':
                searcher = SearcherHybridSearch()
                skg = SemanticKnowledgeGraph()
                user_id = user_id._id if user_id else None
                expanded_terms = searcher.expand_query(user_query, user_id=user_id)
                combined_query = f"{user_query} {' '.join(expanded_terms)}"
                search_results = searcher.search(combined_query, user_id=user_id, top_k=50, rerank_k=10)

                movies = [
                    {
                        'movie': result['movie'],
                        'similarity': min(100, result['score'] * 100)
                    }
                    for result in search_results
                ]
                results_count = len(movies)
            else:
                query_terms = query.lower().split()
                q_objects = Q()
                for term in query_terms:
                    q_objects |= (
                        Q(series_title__icontains=term) |
                        Q(genre__icontains=term) |
                        Q(director__icontains=term) |
                        Q(star1__icontains=term) |
                        Q(star2__icontains=term) |
                        Q(star3__icontains=term) |
                        Q(star4__icontains=term) |
                        Q(overview__icontains=term)
                    )
                movies = all_movies.filter(q_objects).distinct()

                if sort_by == 'rating':
                    movies = movies.order_by('-rating')
                elif sort_by == 'year':
                    movies = movies.order_by('-released_year')
                else:
                    movies = movies.order_by('series_title')

                results_count = movies.count()
        else:
            movies = all_movies.order_by('-rating', '-no_of_votes')[:10]
            results_count = len(movies)

        paginator = Paginator(movies, 12)
        movies_page = paginator.get_page(page)

        movie_stats = get_movie_stats()

        skg = SemanticKnowledgeGraph()
        trending_concepts = [item['concept'] for item in skg.get_trending_concepts(days=7, limit=5)]

        user_id = user.id if user else None
        session_id = get_or_create_session_id(request)
        recommendations = get_recommendations(request, all_movies, num_recommendations=5)
        search_suggestions = get_search_based_suggestions(request, user_id, session_id, all_movies, num_suggestions=5)

        context = {
            'query': query,
            'movies': movies_page,
            'results_count': results_count,
            'search_type': search_type,
            'sort_by': sort_by,
            'show_similarity': show_similarity,
            'movie_stats': movie_stats,
            'trending_concepts': trending_concepts,
            'recommendations': recommendations,
            'search_suggestions': search_suggestions,
            'user_authenticated': user and not is_guest,
            'user': user,
            'is_guest': is_guest,
            'is_staff': is_staff,
        }

        return render(request, 'movies/home.html', context)
    except Exception as e:
        logger.error(f"Search view error: {str(e)}")
        messages.error(request, 'An error occurred during search.')
        return redirect('movies:home')

@login_required
def profile_view(request) -> HttpResponse:
    """Handle user profile page."""
    try:
        user = request.user
        search_history = SearchHistory.objects.filter(user=user).order_by('-timestamp')[:10]
        ratings = Rating.objects.filter(user=user).select_related('movie').order_by('-rating')[:10]
        skg = SemanticKnowledgeGraph()
        trending_concepts = skg.get_trending_concepts()

        return render(request, 'movies/profile.html', {
            'search_history': search_history,
            'ratings': ratings,
            'trending_concepts': trending_concepts
        })
    except Exception as e:
        logger.error(f"Profile view error: {str(e)}")
        messages.error(request, 'Unable to load profile.')
        return redirect('movies:home')

@require_GET
def health_check(request) -> JsonResponse:
    """Check the health of critical services."""
    status = {'status': 'ok', 'services': {}}

    # Check database
    try:
        connections['default'].cursor()
        status['services']['database'] = 'ok'
    except Exception as e:
        status['services']['database'] = f'error: {str(e)}'
        status['status'] = 'error'

    # Check Redis
    try:
        redis_client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=0,
            decode_responses=False,  # Should be False since we're handling binary data
            socket_timeout=5  # Add timeout
        )
        redis_client.ping()
        status['services']['redis'] = 'ok'
    except redis.RedisError as e:
        status['services']['redis'] = f'error: {str(e)}'
        status['status'] = 'error'

    # Check Neo4j
    try:
        driver = GraphDatabase.driver(settings.NEO4J_URI)
        with driver.session() as session:
            session.run("MATCH () RETURN 1 LIMIT 1")
        status['services']['neo4j'] = 'ok'
    except Exception as e:
        status['services']['neo4j'] = f'error: {str(e)}'
        status['status'] = 'error'

    return JsonResponse(status)