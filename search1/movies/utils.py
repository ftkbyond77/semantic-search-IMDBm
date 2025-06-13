from .models import Movie
from collections import defaultdict

class SemanticKnowledgeGraph:
    def __init__(self):
        self.graph = defaultdict(set)
        self.build_graph()

    def build_graph(self):
        """Build SKG from movies dataset"""
        movies = Movie.objects.all()
        for movie in movies:
            genres = [g.strip() for g in movie.genre.split(',')]
            stars = [movie.star1, movie.star2]
            if movie.star3:
                stars.append(movie.star3)
            if movie.star4:
                stars.append(movie.star4)
            
            # Add relationships
            for genre in genres:
                self.graph[movie.director].add(genre)
                for star in stars:
                    self.graph[star].add(genre)
                    self.graph[genre].add(star)
                self.graph[genre].add(movie.director)
            
            for star in stars:
                self.graph[movie.director].add(star)
                self.graph[star].add(movie.director)

    def expand_query(self, query):
        """Expand query using SKG relationships"""
        terms = query.lower().split()
        expanded = set()
        
        for term in terms:
            if term in self.graph:
                related = self.graph[term]
                expanded.update(related)
        
        return list(expanded)[:5]  # Limit to top 5 related terms