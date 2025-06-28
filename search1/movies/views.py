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
import logging
import numpy as np
from scipy.sparse.linalg import svds
import redis
import pickle
import zlib
from neo4j import GraphDatabase
import faiss
import os
from transformers import AutoTokenizer, AutoModel
import dotenv

dotenv.load_dotenv()

logger = logging.getLogger(__name__)

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

def get_neo4j_driver():
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        return driver
    except Exception as e:
        logger.error(f"Neo4j driver initialization error: {e}")
        return None

def get_trending_concepts(driver):
    if not driver:
        return ['Action', 'Drama', 'Sci-Fi']
    try:
        with driver.session() as session:
            result = session.run("MATCH (m:Movie) "
                                "UNWIND split(m.genre, ',') AS genre "
                                "RETURN trim(genre) AS genre, count(*) AS count "
                                "ORDER BY count DESC LIMIT 5")
            return [record['genre'] for record in result]
    except Exception as e:
        logger.error(f"Trending concepts error: {e}")
        return ['Action', 'Drama', 'Sci-Fi']

def get_client_ip(request) -> str:
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR', '')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR', '')
    return ip

def get_or_create_session_id(request) -> str:
    if not request.session.session_key:
        request.session.save()
        logger.info("Created new session key")
    return request.session.session_key

def save_search_history(request, query: str, search_type: str, results_count: int) -> None:
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
    try:
        if ratings_matrix.size == 0 or np.all(ratings_matrix == 0):
            return np.zeros(ratings_matrix.shape), None, None, None
        U, sigma, Vt = svds(ratings_matrix, k=k)
        sigma = np.diag(sigma)
        predicted_ratings = np.dot(np.dot(U, sigma), Vt)
        logger.debug(f"Predicted ratings shape: {predicted_ratings.shape}")
        return predicted_ratings, U, sigma, Vt
    except Exception as e:
        logger.error(f"SVD error: {str(e)}")
        return np.zeros_like(ratings_matrix), None, None, None

def get_recommendations(request, movies_queryset: QuerySet, num_recommendations: int = 5) -> List[Movie]:
    logger.info("Generating personalized recommendations")
    movies_queryset = movies_queryset.select_related().prefetch_related('user_ratings')
    user_id = request.user.id if request.user.is_authenticated else None
    session_id = get_or_create_session_id(request)
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
    if user_id:
        frequent_searches = SearchHistory.objects.filter(user_id=user_id).values('query').annotate(
            count=Count('query')).order_by('-count')[:5]
    else:
        frequent_searches = SearchHistory.objects.filter(session_id=session_id).values('query').annotate(
            count=Count('query')).order_by('-count')[:5]

    logger.debug(f"Frequent searches: {[s['query'] for s in frequent_searches]}")
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
        movie_scores = []
        for movie in candidate_movies:
            rating_boost = (movie.rating or 3.0) / 5.0
            score = 0.6 + 0.2 * rating_boost
            movie_scores.append((movie, score))
        movie_scores.sort(key=lambda x: x[1], reverse=True)
        recommendations = [movie for movie, _ in movie_scores[:num_recommendations]]

    if len(recommendations) < num_recommendations and user_id:
        users = list(set(Rating.objects.values_list('user_id', flat=True)))
        movies_with_ratings = list(set(Rating.objects.values_list('movie_id', flat=True)))
        if users and movies_with_ratings:
            ratings_matrix = np.zeros((len(users), len(movies_with_ratings)))
            for rating in Rating.objects.all():
                user_idx = users.index(rating.user_id)
                movie_idx = movies_with_ratings.index(rating.movie_id)
                ratings_matrix[user_idx, movie_idx] = rating.rating
            predicted_ratings, _, _, _ = matrix_factorization(ratings_matrix, users, movies_with_ratings)
            user_index = users.index(user_id) if user_id in users else -1
            if user_index != -1:
                user_predicted_ratings = predicted_ratings[user_index]
                movie_ratings = []
                for idx, movie_id in enumerate(movies_with_ratings):
                    movie = movies_queryset.filter(id=movie_id).first()
                    if movie and not Rating.objects.filter(user_id=user_id, movie_id=movie_id).exists():
                        movie_ratings.append((movie, user_predicted_ratings[idx]))
                movie_ratings.sort(key=lambda x: x[1], reverse=True)
                additional_recommendations = [
                    movie for movie, _ in movie_ratings[:num_recommendations - len(recommendations)]
                ]
                recommendations.extend(additional_recommendations)

    if len(recommendations) < num_recommendations:
        popular_movies = movies_queryset.order_by(
            '-rating', '-no_of_votes'
        ).exclude(
            id__in=[movie.id for movie in recommendations]
        )[:num_recommendations - len(recommendations)]
        recommendations.extend(popular_movies)

    try:
        serialized = pickle.dumps(recommendations)
        compressed = zlib.compress(serialized)
        redis_client.setex(redis_key, 3600, compressed)
        logger.info(f"Cached recommendations: {[m.series_title.encode('utf-8', errors='replace').decode('utf-8') for m in recommendations]}")
    except Exception as e:
        logger.error(f"Error caching recommendations: {e}")

    return recommendations

def get_search_based_suggestions(request, user_id: Optional[int], session_id: str, movies_queryset: QuerySet, num_suggestions: int = 5) -> List[Movie]:
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
        q_objects = Q(
            Q(series_title__icontains=query) |
            Q(genre__icontains=query) |
            Q(director__icontains=query) |
            Q(star1__icontains=query) |
            Q(star2__icontains=query) |
            Q(star3__icontains=query) |
            Q(star4__icontains=query)
        )
        matches = movies_queryset.filter(q_objects)
        if user_id:
            matches = matches.exclude(
                id__in=Rating.objects.filter(user_id=user_id).values_list('movie_id', flat=True)
            )
        matches = matches[:num_suggestions]
        suggestions.extend(matches)

    seen = set()
    unique_suggestions = []
    for movie in suggestions:
        if movie.id not in seen:
            unique_suggestions.append(movie)
            seen.add(movie.id)
    suggestions = unique_suggestions[:num_suggestions]
    logger.debug(f"Search suggestions: {[s.series_title.encode('utf-8', errors='replace').decode('utf-8') for s in suggestions]}")
    return suggestions

def get_movie_stats() -> Dict[str, Any]:
    try:
        return {
            'total_movies': Movie.objects.count(),
            'avg_rating': Movie.objects.aggregate(avg_rating=Avg('rating'))['avg_rating'] or 0.0,
            'top_genres': Movie.objects.values('genre').annotate(count=Count('genre')).order_by('-count')[:5],
            'recent_movies': Movie.objects.filter(released_year__gte=timezone.now().year - 5).count(),
            'total_ratings': Rating.objects.count(),
        }
    except Exception as e:
        logger.error(f"Error getting movie stats: {str(e)}")
        return {}

def clean_text(text):
    if not text or isinstance(text, float):
        return ""
    try:
        return text.encode('utf-8', errors='replace').decode('utf-8')
    except Exception:
        return text.encode('ascii', errors='replace').decode('ascii')

def home(request) -> HttpResponse:
    query = request.GET.get('q', '').strip()
    search_type = request.GET.get('search_type', 'keyword')
    sort_by = request.GET.get('sort', 'rating')
    page = request.GET.get('page', 1)
    is_guest = request.GET.get('guest', 'false') == 'true'
    user_authenticated = request.user.is_authenticated and not is_guest

    context = {
        'query': query,
        'search_type': search_type,
        'sort_by': sort_by,
        'is_guest': is_guest,
        'user_authenticated': user_authenticated,
        'is_staff': request.user.is_staff if user_authenticated else False,
        'show_similarity': search_type == 'semantic',
    }

    try:
        movie_stats = {
            'total_movies': Movie.objects.count(),
            'avg_rating': Movie.objects.aggregate(Avg('rating'))['rating__avg'] or 0,
            'recent_movies': Movie.objects.filter(released_year__gte=2020).count(),
            'total_ratings': Movie.objects.aggregate(total=Count('user_ratings'))['total'] or 0,
            'top_genres': Movie.objects.values('genre').annotate(count=Count('id')).order_by('-count')[:1],
        }
        context['movie_stats'] = movie_stats
        logger.info(f"Movie stats: {movie_stats}")
    except Exception as e:
        logger.error(f"Movie stats error: {e}")
        context['movie_stats'] = {}

    try:
        driver = get_neo4j_driver()
        context['trending_concepts'] = get_trending_concepts(driver)
    except Exception as e:
        logger.error(f"Neo4j connection error: {e}")
        context['trending_concepts'] = ['Action', 'Drama', 'Sci-Fi']

    session_key = get_or_create_session_id(request)
    logger.info(f"Session key: {session_key}, Query: {query}, Is Guest: {is_guest}")
    try:
        if user_authenticated:
            suggestions = SearchHistory.objects.filter(user=request.user).values('query').annotate(count=Count('query')).order_by('-count')[:5]
            suggestion_queries = [clean_text(s['query']) for s in suggestions]
            context['search_suggestions'] = Movie.objects.filter(series_title__in=suggestion_queries)[:5]
        else:
            suggestions = SearchHistory.objects.filter(session_id=session_key).values('query').annotate(count=Count('query')).order_by('-count')[:5]
            suggestion_queries = [clean_text(s['query']) for s in suggestions]
            context['search_suggestions'] = Movie.objects.filter(series_title__in=suggestion_queries)[:5]
    except Exception as e:
        logger.error(f"Search suggestions error: {e}")
        context['search_suggestions'] = []

    try:
        context['recommendations'] = get_recommendations(request, Movie.objects.all(), num_recommendations=3) if not is_guest else []
    except Exception as e:
        logger.error(f"Recommendations error: {e}")
        context['recommendations'] = []

    movies = []
    results_count = 0
    if query:
        try:
            save_search_history(request, query, search_type, results_count=0)
            index_path = os.path.join(settings.SEARCH_INDEX_PATH, 'search_indices.pkl')
            faiss_index_path = os.path.join(settings.SEARCH_INDEX_PATH, 'faiss_index.bin')
            logger.info(f"Loading indices from {index_path} and {faiss_index_path}")
            if not os.path.exists(index_path) or not os.path.exists(faiss_index_path):
                logger.error("Search indices not found")
                raise FileNotFoundError("Search indices not found")

            with open(index_path, 'rb') as f:
                indices = pickle.load(f)
                bm25 = indices['bm25']
                documents = indices['documents']
            faiss_index = faiss.read_index(faiss_index_path)
            logger.info(f"Loaded {len(documents)} documents from indices")

            if search_type == 'keyword':
                logger.info(f"Performing keyword search for query: {query}")
                tokenized_query = clean_text(query).lower().split()
                scores = bm25.get_scores(tokenized_query)
                logger.info(f"BM25 scores: min={scores.min()}, max={scores.max()}, non-zero={np.sum(scores > 0)}")
                top_indices = np.argsort(scores)[::-1][:100]
                movie_ids = [documents[i]['id'] for i in top_indices if scores[i] > 0]
                logger.info(f"Found {len(movie_ids)} movie IDs from BM25")
                movies = Movie.objects.filter(id__in=movie_ids)
                if sort_by == 'rating':
                    movies = movies.order_by('-rating')
                elif sort_by == 'year':
                    movies = movies.order_by('-released_year')
                else:
                    movies = movies.order_by('series_title')
            else:
                logger.info(f"Performing semantic search for query: {query}")
                tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
                model = AutoModel.from_pretrained('bert-base-uncased')
                inputs = tokenizer([clean_text(query)], return_tensors='pt', padding=True, truncation=True, max_length=512)
                outputs = model(**inputs)
                query_embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()
                D, I = faiss_index.search(query_embedding, k=100)
                logger.info(f"FAISS distances: min={D[0].min()}, max={D[0].max()}")
                movie_ids = [documents[i]['id'] for i, d in zip(I[0], D[0]) if d < 1e6]
                logger.info(f"Found {len(movie_ids)} movie IDs from FAISS")
                movies = Movie.objects.filter(id__in=movie_ids)
                for i, movie in enumerate(movies):
                    movie.similarity = (1 - D[0][i] / D[0].max()) * 100
            results_count = movies.count()
            logger.info(f"Search returned {results_count} movies")

            SearchHistory.objects.filter(
                session_id=session_key,
                query=query,
                timestamp__gte=timezone.now() - timezone.timedelta(minutes=1)
            ).update(results_count=results_count)
        except Exception as e:
            logger.error(f"Search error ({search_type}): {e}")
            movies = Movie.objects.filter(
                Q(series_title__icontains=clean_text(query)) |
                Q(overview__icontains=clean_text(query)) |
                Q(genre__icontains=clean_text(query))
            ).distinct()
            results_count = movies.count()
            logger.info(f"Fallback search returned {results_count} movies")
    else:
        try:
            movies = Movie.objects.order_by('-rating', '-no_of_votes').filter(rating__isnull=False)[:100]
            results_count = movies.count()
            logger.info(f"Default query returned {results_count} movies")
            if results_count == 0:
                logger.warning("No movies found in default query, trying all movies")
                movies = Movie.objects.all()[:100]
                results_count = movies.count()
                logger.info(f"All movies query returned {results_count} movies")
        except Exception as e:
            logger.error(f"Default movies error: {e}")
            movies = []
            results_count = 0

    try:
        # Clean movie data before rendering
        cleaned_movies = []
        for movie in movies:
            try:
                movie.series_title = clean_text(movie.series_title)
                movie.overview = clean_text(movie.overview)
                movie.genre = clean_text(movie.genre)
                cleaned_movies.append(movie)
            except Exception as e:
                logger.warning(f"Error cleaning movie {movie.id}: {e}")
                continue
        paginator = Paginator(cleaned_movies, 20)
        movies_page = paginator.get_page(page)
        context['movies'] = movies_page
        context['results_count'] = results_count
        logger.info(f"Paginated to page {page} with {len(movies_page)} movies")
    except Exception as e:
        logger.error(f"Pagination error: {e}")
        context['movies'] = []
        context['results_count'] = 0

    logger.info(f"Rendering home page: results_count={results_count}, movies={[clean_text(m.series_title) for m in cleaned_movies[:5]]}")
    return render(request, 'movies/home.html', context)

class RatingForm(forms.Form):
    rating = forms.ChoiceField(
        choices=[(i, f"{i} stars") for i in range(1, 6)],
        widget=forms.Select(attrs={'class': 'rating-select'}),
        label="Your Rating"
    )

def auth_view(request):
    if request.user.is_authenticated:
        return redirect('movies:home')
    return render(request, 'movies/auth.html')

def login_view(request):
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
            return redirect('movies:auth')
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

def logout_view(request) -> HttpResponse:
    logout(request)
    messages.success(request, 'You have been logged out successfully')
    return redirect('movies:auth')

def movie_detail(request, movie_id: int) -> HttpResponse:
    try:
        movie = get_object_or_404(Movie.objects.select_related(), id=movie_id)
        movie.series_title = clean_text(movie.series_title)
        movie.overview = clean_text(movie.overview)
        movie.genre = clean_text(movie.genre)
        ratings = Rating.objects.filter(movie=movie).select_related('user')
        user_rating = None
        rating_form = RatingForm()
        avg_rating = ratings.aggregate(avg=Avg('rating'))['avg'] or 0
        rating_distribution = {i: ratings.filter(rating=i).count() for i in range(1, 6)}
        related_movies = Movie.objects.select_related().filter(
            Q(genre__icontains=movie.genre.split(',')[0].strip()) |
            Q(director__icontains=movie.director)
        ).exclude(id=movie_id)[:6]
        for related_movie in related_movies:
            related_movie.series_title = clean_text(related_movie.series_title)
            related_movie.overview = clean_text(related_movie.overview)
            related_movie.genre = clean_text(related_movie.genre)
        if request.user.is_authenticated:
            user_rating = Rating.objects.filter(movie=movie, user=request.user).first()
            if request.method == 'POST':
                rating_form = RatingForm(request.POST)
                if rating_form.is_valid():
                    rating_value = int(rating_form.cleaned_data['rating'])
                    Rating.objects.update_or_create(
                        user=request.user,
                        movie=movie,
                        defaults={'rating': rating_value, 'timestamp': timezone.now()}
                    )
                    messages.success(request, 'Rating updated successfully!')
                    return redirect('movies:movie_detail', movie_id=movie_id)
        return render(request, 'movies/detail.html', {
            'movie': movie,
            'ratings': ratings,
            'user_rating': user_rating,
            'user_authenticated': request.user.is_authenticated,
            'match_explanation': None,
            'boost_factors': None,
            'rating_form': rating_form,
            'avg_rating': round(avg_rating, 1),
            'rating_distribution': rating_distribution,
            'related_movies': related_movies,
            'total_ratings': ratings.count(),
        })
    except Exception as e:
        logger.error(f"Movie detail error for ID {movie_id}: {e}")
        messages.error(request, 'Unable to load movie details.')
        return redirect('movies:home')

def profile_view(request):
    if not request.user.is_authenticated:
        messages.error(request, 'You must be logged in to view your profile.')
        return redirect('movies:auth')
    try:
        user_ratings = Rating.objects.filter(user=request.user).select_related('movie')
        for rating in user_ratings:
            rating.movie.series_title = clean_text(rating.movie.series_title)
        search_history = SearchHistory.objects.filter(user=request.user).order_by('-timestamp')[:10]
        for search in search_history:
            search.query = clean_text(search.query)
        context = {
            'user_ratings': user_ratings,
            'search_history': search_history,
            'user_authenticated': True,
            'is_staff': request.user.is_staff,
        }
        return render(request, 'movies/profile.html', context)
    except Exception as e:
        logger.error(f"Profile view error: {e}")
        messages.error(request, 'Unable to load profile.')
        return redirect('movies:home')

@staff_member_required
def export_search_data(request) -> HttpResponse:
    try:
        search_history = SearchHistory.objects.select_related('user').all()
        data = [
            {
                'id': search.id,
                'user': search.user.username if search.user else None,
                'session_id': search.session_id,
                'query': clean_text(search.query),
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
    try:
        total_searches = SearchHistory.objects.count()
        keyword_searches = SearchHistory.objects.filter(search_type='keyword').count()
        semantic_searches = SearchHistory.objects.filter(search_type='semantic').count()
        popular_queries = SearchHistory.objects.values('query').annotate(
            count=Count('query')
        ).order_by('-count')[:10]
        for query in popular_queries:
            query['query'] = clean_text(query['query'])
        user_search_counts = SearchHistory.objects.filter(user__isnull=False).values(
            'user__username'
        ).annotate(
            count=Count('user')
        ).order_by('-count')[:10]
        driver = get_neo4j_driver()
        trending_concepts = get_trending_concepts(driver)
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

@require_GET
def health_check(request) -> JsonResponse:
    status = {'status': 'ok', 'services': {}}
    try:
        from django.db import connections
        connections['default'].cursor()
        status['services']['database'] = 'ok'
    except Exception as e:
        status['services']['database'] = f'error: {str(e)}'
        status['status'] = 'error'
    try:
        redis_client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=0,
            decode_responses=False,
            socket_timeout=5
        )
        redis_client.ping()
        status['services']['redis'] = 'ok'
    except redis.RedisError as e:
        status['services']['redis'] = f'error: {str(e)}'
        status['status'] = 'error'
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with driver.session() as session:
            session.run("MATCH () RETURN 1 LIMIT 1")
        status['services']['neo4j'] = 'ok'
    except Exception as e:
        status['services']['neo4j'] = f'error: {str(e)}'
        status['status'] = 'error'
    return JsonResponse(status)
