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
import zlib
from typing import List, Dict, Optional, Any
from django.conf import settings
from django.utils import timezone
from movies.models import Movie

logger = logging.getLogger(__name__)

class HybridSearch:
    """Hybrid search combining BM25 and FAISS for efficient movie search."""
    
    def __init__(self):
        self.bm25: Optional[BM25Okapi] = None
        self.faiss_index: Optional[faiss.Index] = None
        self.documents: List[Dict[str, Any]] = []
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = AutoModel.from_pretrained('bert-base-uncased')
        self.dimension: int = 768  # BERT embedding dimension
        self.redis_client = self._init_redis()
        self.load_indices()

    def _init_redis(self) -> redis.Redis:
        """Initialize Redis client with connection pooling and retry logic."""
        try:
            pool = redis.ConnectionPool(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=0,
                max_connections=100,
                socket_timeout=5,
                socket_connect_timeout=5
            )
            client = redis.Redis(connection_pool=pool, decode_responses=True)
            client.ping()  # Test connection
            return client
        except redis.RedisError as e:
            logger.error(f"Redis connection error: {e}")
            raise

    def load_indices(self) -> None:
        """Load BM25 and FAISS indices from disk, with fallback to rebuilding."""
        try:
            index_path = os.path.join(settings.SEARCH_INDEX_PATH, 'search_indices.pkl')
            faiss_index_path = os.path.join(settings.SEARCH_INDEX_PATH, 'faiss_index.bin')
            
            if os.path.exists(index_path) and os.path.exists(faiss_index_path):
                with open(index_path, 'rb') as f:
                    data = pickle.load(f)
                    self.bm25 = data['bm25']
                    self.documents = data['documents']
                self.faiss_index = faiss.read_index(faiss_index_path)
                logger.info("Loaded BM25 and FAISS indices from disk")
            else:
                logger.info("Indices not found, rebuilding...")
                self.build_indices()
        except Exception as e:
            logger.error(f"Error loading indices: {e}")
            self.build_indices()

    def build_indices(self) -> None:
        """Build BM25 and FAISS indices from movie data."""
        try:
            movies = Movie.objects.select_related().all()
            self.documents = []
            corpus = []

            for movie in movies.iterator():
                text = self._build_movie_text(movie)
                self.documents.append({
                    'id': movie.id,
                    'movie': movie,
                    'text': text
                })
                corpus.append(text.split())

            # Build BM25 index
            self.bm25 = BM25Okapi(corpus)
            logger.info("Built BM25 index")

            # Build FAISS index with IVF for scalability
            embeddings = self.get_embeddings([doc['text'] for doc in self.documents])
            nlist = min(100, len(self.documents))  # Number of clusters
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.faiss_index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, faiss.METRIC_L2)
            self.faiss_index.train(embeddings)
            self.faiss_index.add(embeddings)
            logger.info("Built and trained FAISS IVF index")

            # Save indices
            index_path = os.path.join(settings.SEARCH_INDEX_PATH, 'search_indices.pkl')
            faiss_index_path = os.path.join(settings.SEARCH_INDEX_PATH, 'faiss_index.bin')
            os.makedirs(os.path.dirname(index_path), exist_ok=True)
            
            with open(index_path, 'wb') as f:
                pickle.dump({
                    'bm25': self.bm25,
                    'documents': self.documents
                }, f)
            faiss.write_index(self.faiss_index, faiss_index_path)
            logger.info(f"Saved indices to {index_path} and {faiss_index_path}")
        except Exception as e:
            logger.error(f"Error building indices: {e}")
            raise

    def _build_movie_text(self, movie: Movie) -> str:
        """Build text representation of a movie for indexing."""
        text = f"{movie.series_title} {movie.overview or ''} {movie.genre or ''} {movie.director or ''}"
        stars = [s for s in [movie.star1, movie.star2, movie.star3, movie.star4] if s]
        text += f" {' '.join(stars)}"
        if hasattr(movie, 'awards') and movie.awards:
            text += f" {movie.awards}"
        return text

    def get_embeddings(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        """Compute BERT embeddings for texts in batches."""
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            try:
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                outputs = self.model(**inputs)
                batch_embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
                embeddings.append(batch_embeddings)
            except Exception as e:
                logger.error(f"Error computing embeddings for batch {i//batch_size}: {e}")
                embeddings.append(np.zeros((len(batch_texts), self.dimension)))
        return np.vstack(embeddings)

    def search(self, query: str, user_id: Optional[str] = None, top_k: int = 50, rerank_k: int = 10) -> List[Dict[str, Any]]:
        """Perform hybrid search with BM25 and FAISS, with caching."""
        try:
            redis_key = f"search:{query}:{user_id or 'guest'}"
            cached = self.redis_client.get(redis_key)
            if cached:
                logger.debug(f"Cache hit for query: {query}")
                return json.loads(zlib.decompress(cached).decode('utf-8'))

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

            # Cache results with compression
            serialized = json.dumps(results, default=str)
            compressed = zlib.compress(serialized.encode('utf-8'))
            self.redis_client.setex(redis_key, 3600, compressed)
            logger.debug(f"Cached search results for query: {query}")
            return results
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

class SemanticKnowledgeGraph:
    """Manage Neo4j knowledge graph for semantic search and recommendations."""
    
    def __init__(self):
        self.driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            max_connection_lifetime=3600,
            max_connection_pool_size=50
        )
        self.redis_client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=0,
            decode_responses=True
        )

    def close(self) -> None:
        """Close Neo4j driver connection."""
        try:
            self.driver.close()
            logger.info("Closed Neo4j driver connection")
        except Exception as e:
            logger.error(f"Error closing Neo4j driver: {e}")

    def _extract_movie_entities(self, movie: Movie) -> List[str]:
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

    def entity_popularity(self) -> Dict[str, int]:
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

    def expand_query(self, query: str, user_id: Optional[str] = None) -> List[str]:
        """Expand query with related entities from Neo4j."""
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

    def get_recommendation_explanation(self, query: str, expanded_terms: List[str], movie: Movie) -> List[str]:
        """Generate explanations for recommendation relevance."""
        explanation = [f"Matched query: {query}"]
        if expanded_terms:
            explanation.append(f"Related terms: {', '.join(expanded_terms)}")
        explanation.append(f"Movie: {movie.series_title}")
        return explanation

    def get_trending_concepts(self, days: int = 7, limit: int = 5) -> List[Dict[str, Any]]:
        """Get trending concepts based on recent searches."""
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

    def user_preferences(self) -> Dict[str, Dict[str, int]]:
        """Get user preferences based on interactions."""
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

    def update_from_user_interaction(self, user_id: str, movie_id: Optional[str] = None, 
                                  interaction_type: str = 'search', rating_value: Optional[int] = None) -> None:
        """Update knowledge graph with user interactions."""
        try:
            with self.driver.session() as session:
                timestamp = timezone.now().isoformat()
                interaction_id = f"{interaction_type}_{user_id}_{movie_id or 'na'}_{timestamp}"
                
                if interaction_type == 'search':
                    session.run(
                        "MERGE (u:User {id: $user_id}) "
                        "MERGE (s:Search {id: $search_id, timestamp: $timestamp, type: $type}) "
                        "MERGE (u)-[:SEARCHED]->(s) ",
                        user_id=user_id, search_id=interaction_id, timestamp=timestamp, type=interaction_type
                    )
                elif interaction_type == 'view' and movie_id:
                    session.run(
                        "MERGE (u:User {id: $user_id}) "
                        "MERGE (m:Movie {id: $movie_id}) "
                        "MERGE (v:View {id: $view_id, timestamp: $timestamp, type: $type}) "
                        "MERGE (u)-[:VIEWED]->(v)-[:MENTIONS]->(m) ",
                        user_id=user_id, movie_id=movie_id, view_id=interaction_id, timestamp=timestamp, type=interaction_type
                    )
                elif interaction_type == 'rating' and movie_id and rating_value is not None:
                    session.run(
                        "MERGE (u:User {id: $user_id}) "
                        "MERGE (m:Movie {id: $movie_id}) "
                        "MERGE (r:Rating {id: $rating_id, timestamp: $timestamp, type: $type, value: $rating_value}) "
                        "MERGE (u)-[:RATED]->(r)-[:MENTIONS]->(m) ",
                        user_id=user_id, movie_id=movie_id, rating_id=interaction_id, timestamp=timestamp, 
                        type=interaction_type, rating_value=rating_value
                    )

                if movie_id:
                    movie = Movie.objects.filter(id=movie_id).first()
                    if movie:
                        entities = self._extract_movie_entities(movie)
                        for entity in entities:
                            session.run(
                                "MERGE (e:Entity {name: $entity}) "
                                "MERGE (s {id: $interaction_id, timestamp: $timestamp, type: $type}) "
                                "MERGE (s)-[:MENTIONS]->(e) ",
                                entity=entity, interaction_id=interaction_id, timestamp=timestamp, type=interaction_type
                            )
        except Exception as e:
            logger.error(f"Interaction update error: {e}")