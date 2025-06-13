from .models import Movie, Rating, SearchHistory
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import json
import re
import math
from django.db.models import Avg, Count, Q
from django.utils import timezone

class SemanticKnowledgeGraph:
    def __init__(self):
        self.graph = defaultdict(lambda: defaultdict(float))  # weighted edges
        self.entity_types = {}  # track entity types (director, actor, genre, etc.)
        self.entity_popularity = defaultdict(int)  # popularity scores
        self.temporal_weights = defaultdict(lambda: defaultdict(float))  # time-based weights
        self.user_preferences = defaultdict(lambda: defaultdict(float))  # user preference patterns
        self.concept_embeddings = {}  # store concept relationships
        self.synonym_map = {}  # handle synonyms and variations
        self.build_graph()

    def build_graph(self):
        """Build comprehensive SKG from movies dataset"""
        print("Building Semantic Knowledge Graph...")
        movies = Movie.objects.all()
        
        # Initialize synonym mapping
        self._build_synonym_map()
        
        # Build basic entity relationships
        self._build_entity_relationships(movies)
        
        # Build temporal relationships
        self._build_temporal_relationships(movies)
        
        # Build user preference patterns
        self._build_user_patterns()
        
        # Build concept hierarchies
        self._build_concept_hierarchies(movies)
        
        # Calculate entity popularity
        self._calculate_entity_popularity(movies)
        
        print(f"SKG built with {len(self.graph)} entities and {sum(len(edges) for edges in self.graph.values())} relationships")

    def _build_synonym_map(self):
        """Build synonym mappings for better query understanding"""
        self.synonym_map = {
            # Genre synonyms
            'sci-fi': 'science fiction',
            'scifi': 'science fiction',
            'rom-com': 'romance',
            'romcom': 'romance',
            'thriller': 'mystery',
            'superhero': 'action',
            'animated': 'animation',
            'kids': 'family',
            'children': 'family',
            
            # Common variations
            'movie': 'film',
            'movies': 'films',
            'show': 'series',
            'tv': 'television',
            
            # Quality descriptors
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
        """Build basic entity relationships with weighted connections"""
        for movie in movies:
            # Process genres
            genres = [self._normalize_entity(g.strip()) for g in movie.genre.split(',')]
            
            # Process cast
            stars = [movie.star1, movie.star2]
            if movie.star3:
                stars.append(movie.star3)
            if movie.star4:
                stars.append(movie.star4)
            stars = [self._normalize_entity(star) for star in stars if star]
            
            # Process director
            director = self._normalize_entity(movie.director)
            
            # Mark entity types
            for genre in genres:
                self.entity_types[genre] = 'genre'
            for star in stars:
                self.entity_types[star] = 'actor'
            self.entity_types[director] = 'director'
            self.entity_types[movie.series_title.lower()] = 'movie'
            
            # Build relationships with weights based on movie quality and popularity
            movie_weight = self._calculate_movie_weight(movie)
            
            # Director-Genre relationships
            for genre in genres:
                self.graph[director][genre] += movie_weight
                self.graph[genre][director] += movie_weight
            
            # Actor-Genre relationships
            for star in stars:
                for genre in genres:
                    self.graph[star][genre] += movie_weight * 0.8  # slightly lower weight
                    self.graph[genre][star] += movie_weight * 0.8
            
            # Actor-Director collaborations
            for star in stars:
                self.graph[star][director] += movie_weight
                self.graph[director][star] += movie_weight
            
            # Actor-Actor co-appearances
            for i, star1 in enumerate(stars):
                for star2 in stars[i+1:]:
                    self.graph[star1][star2] += movie_weight * 0.6
                    self.graph[star2][star1] += movie_weight * 0.6
            
            # Genre-Genre co-occurrence
            for i, genre1 in enumerate(genres):
                for genre2 in genres[i+1:]:
                    self.graph[genre1][genre2] += movie_weight * 0.7
                    self.graph[genre2][genre1] += movie_weight * 0.7

    def _build_temporal_relationships(self, movies):
        """Build time-based relationship weights"""
        current_year = datetime.now().year
        
        for movie in movies:
            if movie.released_year:
                # Recent movies get higher temporal weights
                years_ago = current_year - movie.released_year
                temporal_factor = max(0.1, 1.0 - (years_ago / 50))  # decay over 50 years
                
                entities = self._extract_movie_entities(movie)
                for entity in entities:
                    self.temporal_weights[entity]['recency'] = max(
                        self.temporal_weights[entity]['recency'], 
                        temporal_factor
                    )

    def _build_user_patterns(self):
        """Analyze user behavior patterns to enhance relationships"""
        # Get user rating patterns
        ratings = Rating.objects.select_related('movie', 'user').all()
        
        for rating in ratings:
            if rating.rating >= 4:  # High ratings indicate preference
                movie = rating.movie
                entities = self._extract_movie_entities(movie)
                
                for entity in entities:
                    self.user_preferences[rating.user.id][entity] += rating.rating / 5.0
        
        # Get search patterns
        recent_searches = SearchHistory.objects.filter(
            timestamp__gte=timezone.now() - timedelta(days=30)
        )
        
        for search in recent_searches:
            query_terms = self._extract_query_terms(search.query)
            for term in query_terms:
                if search.user_id:
                    self.user_preferences[search.user_id][term] += 0.5

    def _build_concept_hierarchies(self, movies):
        """Build hierarchical concept relationships"""
        # Genre hierarchies
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
                self.graph[child][parent] += 0.8  # slightly lower reverse weight
        
        # Era-based concepts
        for movie in movies:
            if movie.released_year:
                era = self._get_movie_era(movie.released_year)
                movie_title = movie.series_title.lower()
                self.graph[movie_title][era] += 1.0
                self.graph[era][movie_title] += 1.0

    def _calculate_entity_popularity(self, movies):
        """Calculate popularity scores for entities"""
        for movie in movies:
            entities = self._extract_movie_entities(movie)
            movie_popularity = movie.no_of_votes / 1000.0  # normalize votes
            
            for entity in entities:
                self.entity_popularity[entity] += movie_popularity

    def _calculate_movie_weight(self, movie):
        """Calculate weight for a movie based on quality indicators"""
        base_weight = 1.0
        
        # Rating boost
        rating_boost = movie.rating / 10.0  # normalize to 0-1
        
        # Popularity boost (log scale to prevent extreme values)
        popularity_boost = math.log1p(movie.no_of_votes) / 15.0
        
        # Meta score boost if available
        meta_boost = (movie.meta_score / 100.0) if movie.meta_score else 0.5
        
        return base_weight + rating_boost + popularity_boost + meta_boost

    def _extract_movie_entities(self, movie):
        """Extract all entities from a movie"""
        entities = []
        
        # Add genres
        genres = [self._normalize_entity(g.strip()) for g in movie.genre.split(',')]
        entities.extend(genres)
        
        # Add cast and director
        cast = [movie.star1, movie.star2]
        if movie.star3:
            cast.append(movie.star3)
        if movie.star4:
            cast.append(movie.star4)
        
        entities.extend([self._normalize_entity(person) for person in cast if person])
        entities.append(self._normalize_entity(movie.director))
        
        return entities

    def _extract_query_terms(self, query):
        """Extract and normalize terms from search query"""
        # Remove punctuation and convert to lowercase
        cleaned = re.sub(r'[^\w\s]', ' ', query.lower())
        terms = [self._normalize_entity(term) for term in cleaned.split() if len(term) > 2]
        return terms

    def _normalize_entity(self, entity):
        """Normalize entity names for consistency"""
        if not entity:
            return ""
        
        normalized = entity.lower().strip()
        # Apply synonym mapping
        return self.synonym_map.get(normalized, normalized)

    def _get_movie_era(self, year):
        """Classify movie into era categories"""
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
        """Enhanced query expansion with personalization and context"""
        terms = self._extract_query_terms(query)
        expanded = set()
        expansion_scores = defaultdict(float)
        
        # Basic expansion from graph relationships
        for term in terms:
            if term in self.graph:
                for related_entity, weight in self.graph[term].items():
                    base_score = weight
                    
                    # Apply temporal weighting
                    temporal_score = self.temporal_weights[related_entity].get('recency', 0.5)
                    
                    # Apply user preference weighting if user provided
                    user_score = 1.0
                    if user_id and user_id in self.user_preferences:
                        user_score = self.user_preferences[user_id].get(related_entity, 0.5) + 0.5
                    
                    # Apply popularity weighting
                    popularity_score = min(2.0, self.entity_popularity[related_entity] / 100.0)
                    
                    # Combined score
                    final_score = base_score * temporal_score * user_score * (1 + popularity_score)
                    expansion_scores[related_entity] = final_score
        
        # Add contextual expansions
        self._add_contextual_expansions(terms, expansion_scores)
        
        # Sort by score and return top expansions
        sorted_expansions = sorted(expansion_scores.items(), key=lambda x: x[1], reverse=True)
        expanded = [entity for entity, score in sorted_expansions[:max_expansions]]
        
        print(f"Query '{query}' expanded to: {expanded[:5]}...")  # Show top 5 for debugging
        return expanded

    def _add_contextual_expansions(self, terms, expansion_scores):
        """Add contextual expansions based on query analysis"""
        # Detect query intent
        intent = self._detect_query_intent(terms)
        
        if intent == 'genre_seeking':
            # If user is looking for genres, boost related genres
            for term in terms:
                if self.entity_types.get(term) == 'genre':
                    for related in self.graph[term]:
                        if self.entity_types.get(related) == 'genre':
                            expansion_scores[related] *= 1.5
        
        elif intent == 'actor_focused':
            # If user mentions actors, boost their frequent collaborators
            for term in terms:
                if self.entity_types.get(term) == 'actor':
                    for related in self.graph[term]:
                        if self.entity_types.get(related) in ['actor', 'director']:
                            expansion_scores[related] *= 1.3
        
        elif intent == 'era_seeking':
            # If user mentions time-related terms, boost era-related content
            era_terms = ['old', 'new', 'recent', 'classic', 'modern', 'vintage']
            if any(era_term in terms for era_term in era_terms):
                for era in ['classic', 'modern', '90s', '2000s', '2010s']:
                    if era in self.graph:
                        expansion_scores[era] *= 1.2

    def _detect_query_intent(self, terms):
        """Detect the intent behind the search query"""
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
        """Get detailed information about an entity"""
        entity = self._normalize_entity(entity)
        
        info = {
            'entity': entity,
            'type': self.entity_types.get(entity, 'unknown'),
            'popularity': self.entity_popularity.get(entity, 0),
            'relationships': dict(self.graph.get(entity, {})),
            'temporal_weight': self.temporal_weights.get(entity, {})
        }
        
        return info

    def get_recommendation_explanation(self, query, expanded_terms, movie):
        """Generate explanation for why a movie was recommended"""
        explanations = []
        
        # Check direct matches
        movie_entities = self._extract_movie_entities(movie)
        query_terms = self._extract_query_terms(query)
        
        # Direct matches
        direct_matches = set(query_terms) & set(movie_entities)
        if direct_matches:
            explanations.append(f"Direct match: {', '.join(direct_matches)}")
        
        # Expanded term matches
        expanded_matches = set(expanded_terms) & set(movie_entities)
        if expanded_matches:
            explanations.append(f"Related concepts: {', '.join(list(expanded_matches)[:3])}")
        
        # Quality indicators
        if movie.rating > 8.0:
            explanations.append("High IMDB rating")
        
        if movie.no_of_votes > 100000:
            explanations.append("Popular choice")
        
        return explanations

    def update_from_user_interaction(self, user_id, movie_id, interaction_type, value=1.0):
        """Update graph based on user interactions"""
        try:
            movie = Movie.objects.get(id=movie_id)
            entities = self._extract_movie_entities(movie)
            
            # Update user preferences
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
        """Get trending concepts based on recent search activity"""
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
        """Export graph data for analysis or backup"""
        return {
            'graph': dict(self.graph),
            'entity_types': dict(self.entity_types),
            'entity_popularity': dict(self.entity_popularity),
            'temporal_weights': dict(self.temporal_weights),
            'user_preferences': dict(self.user_preferences),
            'export_timestamp': timezone.now().isoformat()
        }