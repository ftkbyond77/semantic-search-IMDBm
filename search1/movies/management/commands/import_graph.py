from django.core.management.base import BaseCommand
from neo4j import GraphDatabase
from movies.models import Movie
from django.conf import settings

class Command(BaseCommand):
    help = 'Import movie data into Neo4j Knowledge Graph'

    def handle(self, *args, **kwargs):
        uri = settings.NEO4J_URI
        driver = GraphDatabase.driver(uri)

        def create_graph(tx, movie):
            tx.run("""
                MERGE (m:Movie {title: $title, imdb_id: $imdb_id})
                SET m.released_year = $released_year, m.rating = $rating
            """, title=movie.series_title, imdb_id=str(movie.id),
                released_year=movie.released_year, rating=movie.rating)

            if movie.director:
                tx.run("""
                    MERGE (d:Director {name: $name})
                    MERGE (m:Movie {title: $title})
                    MERGE (m)-[:DIRECTED_BY]->(d)
                """, name=movie.director, title=movie.series_title)

            for star in [movie.star1, movie.star2, movie.star3, movie.star4]:
                if star:
                    tx.run("""
                        MERGE (a:Actor {name: $name})
                        MERGE (m:Movie {title: $title})
                        MERGE (a)-[:ACTED_IN]->(m)
                    """, name=star, title=movie.series_title)

            if movie.genre:
                genres = [g.strip() for g in movie.genre.split(',')]
                for genre in genres:
                    tx.run("""
                        MERGE (g:Genre {name: $name})
                        MERGE (m:Movie {title: $title})
                        MERGE (m)-[:HAS_GENRE]->(g)
                    """, name=genre, title=movie.series_title)

            if movie.overview and 'oscar' in movie.overview.lower():
                tx.run("""
                    MERGE (a:Award {name: $name, year: $year})
                    MERGE (m:Movie {title: $title})
                    MERGE (m)-[:WON_AWARD]->(a)
                """, name="Oscar", year=movie.released_year, title=movie.series_title)

        try:
            with driver.session() as session:
                for movie in Movie.objects.all():
                    session.write_transaction(create_graph, movie)
                    self.stdout.write(f"Imported {movie.series_title}")
        except Exception as e:
            self.stderr.write(f"Error importing data: {e}")
            raise

        driver.close()
        self.stdout.write(self.style.SUCCESS("Graph import completed!"))