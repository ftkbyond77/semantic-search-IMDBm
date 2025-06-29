from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from mvapp.models import Movie
from sentence_transformers import SentenceTransformer
import numpy as np
from neo4j import GraphDatabase

def index(request):
    return render(request, 'mvapp/index.html')

class SearchView(APIView):
    def get(self, request):
        query = request.GET.get('q', '')
        model = SentenceTransformer('all-MiniLM-L6-v2')
        query_embedding = model.encode([query])[0]
        
        # Basic search (keyword)
        keyword_results = Movie.objects.filter(series_title__icontains=query) | \
                         Movie.objects.filter(overview__icontains=query)
        
        # Semantic search
        movies = Movie.objects.exclude(embedding__isnull=True)
        semantic_results = []
        for movie in movies:
            movie_embedding = np.array(movie.embedding)
            similarity = np.dot(query_embedding, movie_embedding) / \
                        (np.linalg.norm(query_embedding) * np.linalg.norm(movie_embedding))
            semantic_results.append((movie, similarity))
        semantic_results = sorted(semantic_results, key=lambda x: x[1], reverse=True)[:10]
        
        # Neo4j graph search (example: find movies by director)
        driver = GraphDatabase.driver("bolt://neo4j:7687", auth=("neo4j", "secretpassword"))
        with driver.session() as session:
            graph_results = session.run(
                """
                MATCH (m:Movie)-[:DIRECTED_BY]->(d:Director)
                WHERE toLower(d.name) CONTAINS toLower($query)
                RETURN m.title
                """,
                query=query
            ).values()
        driver.close()
        
        return Response({
            'keyword_results': [m.series_title for m in keyword_results],
            'semantic_results': [m[0].series_title for m in semantic_results],
            'graph_results': [r[0] for r in graph_results]
        })