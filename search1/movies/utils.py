import os
import numpy as np
from rank_bm25 import BM25Okapi
import faiss
from transformers import AutoTokenizer, AutoModel
from neo4j import GraphDatabase
import redis
import pickle
import json
import logging
from django.conf import settings
from django.utils import timezone

logger = logging.getLogger(__name__)

class HybridSearch:
    def __init__(self):
        self.bm25 = None
        self.faiss_index = None
        self.documents = None
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = AutoModel.from_pretrained('bert-base-uncased')
        self.dimension = 768  # BERT embedding dimension
        self.redis_client = redis.Redis(host='redis', port=6379, db=0)
        self.load_indices()

    def load_indices(self):
        try:
            index_path = os.path.join(settings.SEARCH_INDEX_PATH, 'search_indices.pkl')
            faiss_index_path = os.path.join(settings.SEARCH_INDEX_PATH, 'faiss_index.bin')
            with open(index_path, 'rb') as f:
                data = pickle.load(f)
                self.bm25 = data['bm25']
                self.documents = data['documents']
            self.faiss_index = faiss.read_index(faiss_index_path)
        except Exception as e:
            logger.error(f"Error loading indices: {e}")
            self.build_indices()

    def build_indices(self):
        from movies.models import Movie
        movies = Movie.objects.all()
        self.documents = []
        corpus = []

        for movie in movies:
            text = f"{movie.series_title} {movie.overview} {movie.genre} {movie.director} {' '.join([s for s in [movie.star1, movie.star2, movie.star3, movie.star4] if s])}"
            if hasattr(movie, 'awards'):
                text += f" {movie.awards}"
            self.documents.append({
                'id': movie.id,
                'movie': movie,
                'text': text
            })
            corpus.append(text.split())

        self.bm25 = BM25Okapi(corpus)
        embeddings = self.get_embeddings([doc['text'] for doc in self.documents])
        self.faiss_index = faiss.IndexFlatL2(self.dimension)
        self.faiss_index.add(embeddings)

        index_path = os.path.join(settings.SEARCH_INDEX_PATH, 'search_indices.pkl')
        faiss_index_path = os.path.join(settings.SEARCH_INDEX_PATH, 'faiss_index.bin')
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        with open(index_path, 'wb') as f:
            pickle.dump({
                'bm25': self.bm25,
                'documents': self.documents
            }, f)
        faiss.write_index(self.faiss_index, faiss_index_path)

    def get_embeddings(self, texts):
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
        return embeddings

    def search(self, query, user_id=None, top_k=50, rerank_k=10):
        try:
            redis_key = f"search:{query}:{user_id or 'guest'}"
            cached = self.redis_client.get(redis_key)
            if cached:
                return json.loads(cached)

            query_embedding = self.get_embeddings([query])[0]
            bm25_scores = self.bm25.get_scores(query.split())
            _, faiss_indices = self.faiss_index.search(np.array([query_embedding]), top_k)

            combined_scores = {}
            for idx, score in enumerate(bm25_scores):
                combined_scores[idx] = combined_scores.get(idx, 0) + score * 0.5
            for idx in faiss_indices[0]:
                combined_scores[int(idx)] = combined_scores.get(int(idx), 0) + 0.5

            top_indices = sorted(combined_scores, key=combined_scores.get, reverse=True)[:rerank_k]
            results = [
                {
                    'movie': self.documents[idx]['movie'],
                    'score': combined_scores[idx]
                }
                for idx in top_indices
            ]

            self.redis_client.setex(redis_key, 3600, json.dumps(results, default=str))
            return results
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

class SemanticKnowledgeGraph:
    def __init__(self):
        self.driver = GraphDatabase.driver(settings.NEO4J_URI)
        self.redis_client = redis.Redis(host='redis', port=6379, db=0)

    def close(self):
        self.driver.close()

    def _extract_movie_entities(self, movie):
        """Extract entities (genre, director, stars) from a movie."""
        entities = []
        if movie.genre:
            entities.extend([g.strip().lower() for g in movie.genre.split(',') if g.strip()])
        if movie.director:
            entities.append(movie.director.lower())
        for star in [movie.star1, movie.star2, movie.star3, movie.star4]:
            if star:
                entities.append(star.lower())
        if hasattr(movie, 'awards') and movie.awards:
            entities.append(movie.awards.lower())
        return list(set(entities))

    @property
    def entity_popularity(self):
        """Get popularity scores for entities based on search interactions."""
        try:
            redis_key = "entity_popularity"
            cached = self.redis_client.get(redis_key)
            if cached:
                return json.loads(cached)

            with self.driver.session() as session:
                result = session.run(
                    "MATCH (e:Entity)<-[:MENTIONS]-(s:Search) "
                    "RETURN e.name, count(s) as count "
                    "ORDER BY count DESC"
                )
                popularity = {record["e.name"]: record["count"] for record in result}
                self.redis_client.setex(redis_key, 3600, json.dumps(popularity))
                return popularity
        except Exception as e:
            logger.error(f"Entity popularity error: {e}")
            return {}

    def expand_query(self, query, user_id=None):
        try:
            redis_key = f"expand:{query}:{user_id or 'guest'}"
            cached = self.redis_client.get(redis_key)
            if cached:
                return json.loads(cached)

            with self.driver.session() as session:
                result = session.run(
                    "MATCH (e:Entity) WHERE toLower(e.name) CONTAINS toLower($query) "
                    "RETURN e.name LIMIT 5",
                    query=query
                )
                entities = [record["e.name"] for record in result]
                self.redis_client.setex(redis_key, 3600, json.dumps(entities))
                return entities
        except Exception as e:
            logger.error(f"Query expansion error: {e}")
            return []

    def get_recommendation_explanation(self, query, expanded_terms, movie):
        explanation = [f"Matched query: {query}"]
        if expanded_terms:
            explanation.append(f"Related terms: {', '.join(expanded_terms)}")
        explanation.append(f"Movie: {movie.series_title}")
        return explanation

    def get_trending_concepts(self, days=7, limit=5):
        try:
            redis_key = f"trending:{days}:{limit}"
            cached = self.redis_client.get(redis_key)
            if cached:
                return json.loads(cached)

            with self.driver.session() as session:
                result = session.run(
                    "MATCH (e:Entity)<-[:SEARCHED]-(s:Search) "
                    "WHERE s.timestamp >= datetime() - duration('P' + $days + 'D') "
                    "RETURN e.name, count(s) as count "
                    "ORDER BY count DESC LIMIT $limit",
                    days=str(days), limit=limit
                )
                concepts = [{"concept": record["e.name"], "count": record["count"]} for record in result]
                self.redis_client.setex(redis_key, 3600, json.dumps(concepts))
                return concepts
        except Exception as e:
            logger.error(f"Trending concepts error: {e}")
            return []

    @property
    def user_preferences(self):
        try:
            with self.driver.session() as session:
                result = session.run(
                    "MATCH (u:User)-[:SEARCHED | :RATED | :VIEWED]->(s)-[:MENTIONS]->(e:Entity) "
                    "RETURN u.id, e.name, count(s) as count"
                )
                preferences = {}
                for record in result:
                    user_id = record["u.id"]
                    entity = record["e.name"]
                    count = record["count"]
                    if user_id not in preferences:
                        preferences[user_id] = {}
                    preferences[user_id][entity] = count
                return preferences
        except Exception as e:
            logger.error(f"User preferences error: {e}")
            return {}

    def update_from_user_interaction(self, user_id, movie_id=None, interaction_type='search', rating_value=None):
        """Update the knowledge graph with user interactions."""
        try:
            with self.driver.session() as session:
                timestamp = timezone.now().isoformat()
                if interaction_type == 'search':
                    # Create or update Search node
                    session.run(
                        "MERGE (u:User {id: $user_id}) "
                        "MERGE (s:Search {id: $search_id, timestamp: $timestamp, type: $type}) "
                        "MERGE (u)-[:SEARCHED]->(s) ",
                        user_id=user_id, search_id=f"search_{user_id}_{timestamp}", timestamp=timestamp, type=interaction_type
                    )
                elif interaction_type == 'view' and movie_id:
                    # Link user to movie view
                    session.run(
                        "MERGE (u:User {id: $user_id}) "
                        "MERGE (m:Movie {id: $movie_id}) "
                        "MERGE (v:View {id: $view_id, timestamp: $timestamp, type: $type}) "
                        "MERGE (u)-[:VIEWED]->(v)-[:MENTIONS]->(m) ",
                        user_id=user_id, movie_id=movie_id, view_id=f"view_{user_id}_{movie_id}_{timestamp}", timestamp=timestamp, type=interaction_type
                    )
                elif interaction_type == 'rating' and movie_id and rating_value is not None:
                    # Link user to movie rating
                    session.run(
                        "MERGE (u:User {id: $user_id}) "
                        "MERGE (m:Movie {id: $movie_id}) "
                        "MERGE (r:Rating {id: $rating_id, timestamp: $timestamp, type: $type, value: $rating_value}) "
                        "MERGE (u)-[:RATED]->(r)-[:MENTIONS]->(m) ",
                        user_id=user_id, movie_id=movie_id, rating_id=f"rating_{user_id}_{movie_id}_{timestamp}", timestamp=timestamp, type=interaction_type, rating_value=rating_value
                    )

                # Update entities for movie if applicable
                if movie_id:
                    movie = Movie.objects.filter(id=movie_id).first()
                    if movie:
                        entities = self._extract_movie_entities(movie)
                        for entity in entities:
                            session.run(
                                "MERGE (e:Entity {name: $entity}) "
                                "MERGE (s {id: $interaction_id, timestamp: $timestamp, type: $type}) "
                                "MERGE (s)-[:MENTIONS]->(e) ",
                                entity=entity, interaction_id=f"{interaction_type}_{user_id}_{movie_id}_{timestamp}", timestamp=timestamp, type=interaction_type
                            )
        except Exception as e:
            logger.error(f"Interaction update error: {e}")