from .models import Movie, Rating, SearchHistory
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import re
import math
from django.db.models import Avg, Count, Q
from django.utils import timezone
from django.core.cache import cache
import spacy
nlp = spacy.load('en_core_web_sm')

def default_float_dict():
    """Factory function for defaultdict with float values"""
    return defaultdict(float)

class SemanticKnowledgeGraph:
    def __init__(self):
        self.graph = defaultdict(default_float_dict)  # Use factory function
        self.entity_types = {}
        self.entity_popularity = defaultdict(int)
        self.temporal_weights = defaultdict(default_float_dict)
        self.user_preferences = defaultdict(default_float_dict)
        self.concept_embeddings = {}
        self.synonym_map = {}
        self.build_graph()

    def build_graph(self):
        cached_graph = cache.get('semantic_knowledge_graph')
        if cached_graph:
            self.graph = defaultdict(default_float_dict, {k: defaultdict(float, v) for k, v in cached_graph[0].items()})
            self.entity_types = cached_graph[1]
            self.entity_popularity = defaultdict(int, cached_graph[2])
            print("Loaded SKG from cache")
            return
        
        print("Building Semantic Knowledge Graph...")
        movies = Movie.objects.all()
        
        self._build_synonym_map()
        self._build_entity_relationships(movies)
        self._build_temporal_relationships(movies)
        self._build_user_patterns()
        self._build_concept_hierarchies(movies)
        self._calculate_entity_popularity(movies)
        
        try:
            # Convert defaultdicts to regular dicts for caching
            cache.set('semantic_knowledge_graph', (
                {k: dict(v) for k, v in self.graph.items()},
                dict(self.entity_types),
                dict(self.entity_popularity)
            ), timeout=3600)
            print("Cached SKG successfully")
        except Exception as e:
            print(f"Failed to cache SKG: {e}")
        
        print(f"SKG built with {len(self.graph)} entities and {sum(len(edges) for edges in self.graph.values())} relationships")

    def _build_synonym_map(self):
        self.synonym_map = {
            'sci-fi': 'science fiction',
            'scifi': 'science fiction',
            'rom-com': 'romance',
            'romcom': 'romance',
            'thriller': 'mystery',
            'superhero': 'action',
            'animated': 'animation',
            'kids': 'family',
            'children': 'family',
            'movie': 'film',
            'movies': 'films',
            'show': 'series',
            'tv': 'television',
            'good': 'high-rated',
            'best': 'top-rated',
            'popular': 'well-known',
            'famous': 'well-known',
            'classic': 'timeless',
            'old': 'vintage',
            'new': 'recent',
            'latest': 'recent'
        }

    def _build_entity_relationships(self, movies):
        for movie in movies:
            genres = [self._normalize_entity(g.strip()) for g in movie.genre.split(',') if movie.genre]
            stars = [s for s in [movie.star1, movie.star2, movie.star3, movie.star4] if s]
            stars = [self._normalize_entity(star) for star in stars]
            director = self._normalize_entity(movie.director) if movie.director else ''
            
            for genre in genres:
                self.entity_types[genre] = 'genre'
            for star in stars:
                self.entity_types[star] = 'actor'
            if director:
                self.entity_types[director] = 'director'
            self.entity_types[movie.series_title.lower()] = 'movie'
            
            movie_weight = self._calculate_movie_weight(movie)
            
            if director:
                for genre in genres:
                    self.graph[director][genre] += movie_weight
                    self.graph[genre][director] += movie_weight
            
            for star in stars:
                for genre in genres:
                    self.graph[star][genre] += movie_weight * 0.8
                    self.graph[genre][star] += movie_weight * 0.8
            
            for star in stars:
                if director:
                    self.graph[star][director] += movie_weight
                    self.graph[director][star] += movie_weight
            
            for i, star1 in enumerate(stars):
                for star2 in stars[i+1:]:
                    self.graph[star1][star2] += movie_weight * 0.6
                    self.graph[star2][star1] += movie_weight * 0.6
            
            for i, genre1 in enumerate(genres):
                for genre2 in genres[i+1:]:
                    self.graph[genre1][genre2] += movie_weight * 0.7
                    self.graph[genre2][genre1] += movie_weight * 0.7

    def _build_temporal_relationships(self, movies):
        current_year = datetime.now().year
        for movie in movies:
            if movie.released_year:
                years_ago = current_year - movie.released_year
                temporal_factor = max(0.1, 1.0 - (years_ago / 50))
                entities = self._extract_movie_entities(movie)
                for entity in entities:
                    self.temporal_weights[entity]['recency'] = max(
                        self.temporal_weights[entity]['recency'], 
                        temporal_factor
                    )

    def _build_user_patterns(self):
        ratings = Rating.objects.select_related('movie', 'user').all()
        for rating in ratings:
            if rating.rating >= 4:
                movie = rating.movie
                entities = self._extract_movie_entities(movie)
                for entity in entities:
                    self.user_preferences[rating.user.id][entity] += rating.rating / 5.0
        
        recent_searches = SearchHistory.objects.filter(
            timestamp__gte=timezone.now() - timedelta(days=30)
        )
        for search in recent_searches:
            query_terms = self._extract_query_terms(search.query)
            for term in query_terms:
                if search.user_id:
                    self.user_preferences[search.user_id][term] += 0.5

    def _build_concept_hierarchies(self, movies):
        genre_hierarchy = {
            'action': ['adventure', 'thriller', 'war', 'western'],
            'drama': ['biography', 'history', 'romance'],
            'comedy': ['family', 'animation'],
            'horror': ['thriller', 'mystery'],
            'science fiction': ['fantasy', 'adventure']
        }
        for parent, children in genre_hierarchy.items():
            for child in children:
                self.graph[parent][child] += 1.0
                self.graph[child][parent] += 0.8
        
        for movie in movies:
            if movie.released_year:
                era = self._get_movie_era(movie.released_year)
                movie_title = movie.series_title.lower()
                self.graph[movie_title][era] += 1.0
                self.graph[era][movie_title] += 1.0

    def _calculate_entity_popularity(self, movies):
        for movie in movies:
            entities = self._extract_movie_entities(movie)
            movie_popularity = movie.no_of_votes / 1000.0
            for entity in entities:
                self.entity_popularity[entity] += movie_popularity

    def _calculate_movie_weight(self, movie):
        base_weight = 1.0
        rating_boost = movie.rating / 10.0
        popularity_boost = math.log1p(movie.no_of_votes) / 15.0
        meta_boost = (movie.meta_score / 100.0) if movie.meta_score else 0.5
        return base_weight + rating_boost + popularity_boost + meta_boost

    def _extract_movie_entities(self, movie):
        entities = []
        genres = [self._normalize_entity(g.strip()) for g in movie.genre.split(',') if movie.genre]
        entities.extend(genres)
        cast = [s for s in [movie.star1, movie.star2, movie.star3, movie.star4] if s]
        entities.extend([self._normalize_entity(person) for person in cast])
        if movie.director:
            entities.append(self._normalize_entity(movie.director))
        return entities

    def _extract_query_terms(self, query):
        cleaned = re.sub(r'[^\w\s]', ' ', query.lower())
        terms = [self._normalize_entity(term) for term in cleaned.split() if len(term) > 2]
        return terms

    def _normalize_entity(self, entity):
        if not entity:
            return ""
        normalized = entity.lower().strip()
        return self.synonym_map.get(normalized, normalized)

    def _get_movie_era(self, year):
        if year >= 2020:
            return "modern"
        elif year >= 2010:
            return "2010s"
        elif year >= 2000:
            return "2000s"
        elif year >= 1990:
            return "90s"
        elif year >= 1980:
            return "80s"
        elif year >= 1970:
            return "70s"
        else:
            return "classic"

    def expand_query(self, query, user_id=None, max_expansions=8):
        terms = self._extract_query_terms(query)
        expanded = set()
        expansion_scores = defaultdict(float)
        
        for term in terms:
            if term in self.graph:
                for related_entity, weight in self.graph[term].items():
                    base_score = weight
                    temporal_score = self.temporal_weights[related_entity].get('recency', 0.5)
                    user_score = 1.0
                    if user_id and user_id in self.user_preferences:
                        user_score = self.user_preferences[user_id].get(related_entity, 0.5) + 0.5
                    popularity_score = min(2.0, self.entity_popularity[related_entity] / 100.0)
                    final_score = base_score * temporal_score * user_score * (1 + popularity_score)
                    expansion_scores[related_entity] = final_score
        
        self._add_contextual_expansions(terms, expansion_scores)
        sorted_expansions = sorted(expansion_scores.items(), key=lambda x: x[1], reverse=True)
        expanded = [entity for entity, score in sorted_expansions[:max_expansions]]
        
        print(f"Query '{query}' expanded to: {expanded[:5]}...")
        return expanded

    def _add_contextual_expansions(self, terms, expansion_scores):
        intent = self._detect_query_intent(terms)
        if intent == 'genre_seeking':
            for term in terms:
                if self.entity_types.get(term) == 'genre':
                    for related in self.graph[term]:
                        if self.entity_types.get(related) == 'genre':
                            expansion_scores[related] *= 1.5
        elif intent == 'actor_focused':
            for term in terms:
                if self.entity_types.get(term) == 'actor':
                    for related in self.graph[term]:
                        if self.entity_types.get(related) in ['actor', 'director']:
                            expansion_scores[related] *= 1.3
        elif intent == 'era_seeking':
            era_terms = ['old', 'new', 'recent', 'classic', 'modern', 'vintage']
            if any(era_term in terms for era_term in era_terms):
                for era in ['classic', 'modern', '90s', '2000s', '2010s']:
                    if era in self.graph:
                        expansion_scores[era] *= 1.2

    def _detect_query_intent(self, terms):
        genre_count = sum(1 for term in terms if self.entity_types.get(term) == 'genre')
        actor_count = sum(1 for term in terms if self.entity_types.get(term) == 'actor')
        director_count = sum(1 for term in terms if self.entity_types.get(term) == 'director')
        
        if genre_count > actor_count and genre_count > director_count:
            return 'genre_seeking'
        elif actor_count > genre_count:
            return 'actor_focused'
        elif any(term in ['old', 'new', 'recent', 'classic'] for term in terms):
            return 'era_seeking'
        else:
            return 'general'

    def get_entity_info(self, entity):
        entity = self._normalize_entity(entity)
        info = {
            'entity': entity,
            'type': self.entity_types.get(entity, 'unknown'),
            'popularity': self.entity_popularity.get(entity, 0),
            'relationships': dict(self.graph.get(entity, {})),
            'temporal_weight': dict(self.temporal_weights.get(entity, {}))
        }
        return info

    def get_recommendation_explanation(self, query, expanded_terms, movie):
        explanations = []
        movie_entities = self._extract_movie_entities(movie)
        query_terms = self._extract_query_terms(query)
        
        direct_matches = set(query_terms) & set(movie_entities)
        if direct_matches:
            explanations.append(f"Direct match: {', '.join(direct_matches)}")
        
        expanded_matches = set(expanded_terms) & set(movie_entities)
        if expanded_matches:
            explanations.append(f"Related concepts: {', '.join(list(expanded_matches)[:3])}")
        
        if movie.rating > 8.0:
            explanations.append("High IMDB rating")
        
        if movie.no_of_votes > 100000:
            explanations.append("Popular choice")
        
        return explanations

    def update_from_user_interaction(self, user_id, movie_id, interaction_type, value=1.0):
        try:
            movie = Movie.objects.get(id=movie_id)
            entities = self._extract_movie_entities(movie)
            for entity in entities:
                if interaction_type == 'rating' and value >= 4:
                    self.user_preferences[user_id][entity] += value / 5.0
                elif interaction_type == 'view':
                    self.user_preferences[user_id][entity] += 0.1
                elif interaction_type == 'search':
                    self.user_preferences[user_id][entity] += 0.2
            print(f"Updated SKG from user {user_id} interaction with movie {movie_id}")
        except Movie.DoesNotExist:
            print(f"Movie {movie_id} not found for SKG update")

    def get_trending_concepts(self, days=7, limit=10):
        recent_searches = SearchHistory.objects.filter(
            timestamp__gte=timezone.now() - timedelta(days=days)
        )
        concept_counts = Counter()
        for search in recent_searches:
            terms = self._extract_query_terms(search.query)
            for term in terms:
                concept_counts[term] += 1
        trending = []
        for concept, count in concept_counts.most_common(limit):
            trend_info = {
                'concept': concept,
                'search_count': count,
                'type': self.entity_types.get(concept, 'keyword'),
                'popularity': self.entity_popularity.get(concept, 0)
            }
            trending.append(trend_info)
        return trending

    def export_graph_data(self):
        return {
            'graph': {k: dict(v) for k, v in self.graph.items()},
            'entity_types': dict(self.entity_types),
            'entity_popularity': dict(self.entity_popularity),
            'temporal_weights': {k: dict(v) for k, v in self.temporal_weights.items()},
            'user_preferences': {k: dict(v) for k, v in self.user_preferences.items()},
            'export_timestamp': timezone.now().isoformat()
        }

from neo4j import GraphDatabase
from django.conf import settings

class Neo4jClient:
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.driver = GraphDatabase.driver(
                kwargs.get('uri', settings.NEO4J_URI)
            )
        return cls._instance
    
    def close(self):
        if self._instance:
            self.driver.close()
            self._instance = None

    def find_movies_by_actor_decade_award(self, actor_name, decade_start, award_name):
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (a:Actor {name: $actor_name})-[:ACTED_IN]->(m:Movie)
                    WHERE m.released_year >= $decade_start AND m.released_year < $decade_start + 10
                    MATCH (m)-[:WON_AWARD]->(aw:Award {name: $award_name})
                    RETURN m.title, m.released_year, m.rating, m.imdb_id
                    ORDER BY m.rating DESC
                """, actor_name=actor_name, decade_start=decade_start, award_name=award_name)
                return [{"title": r["m.title"], "year": r["m.released_year"],
                         "rating": r["m.rating"], "id": r["m.imdb_id"]} for r in result]
        except Exception as e:
            print(f"Neo4j query error: {e}")
            return []

def search_movies_with_graph(query, user_id=None):
    skg = SemanticKnowledgeGraph()
    expanded_terms = skg.expand_query(query, user_id)
    
    client = Neo4jClient()
    doc = nlp(query.lower())
    
    actor = None
    decade = None
    award = None
    
    for term in expanded_terms + [ent.text for ent in doc.ents]:
        if skg.entity_types.get(term, '') == 'actor':
            actor = term
        elif '90s' in term or term == '1990s':
            decade = 1990
        elif term == 'oscar':
            award = 'Oscar'
    
    if actor and decade and award:
        results = client.find_movies_by_actor_decade_award(actor, decade, award)
        client.close()
        return results
    
    client.close()
    return []