import csv
import logging
import os
from django.core.management.base import BaseCommand
from movies.models import Movie
from django.db import transaction
from neo4j import GraphDatabase
import dotenv

dotenv.load_dotenv()

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Import IMDb data from CSV and populate Neo4j'

    def handle(self, *args, **kwargs):
        NEO4J_URI = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
        NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
        NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
        csv_file_path = os.getenv("CSV_FILE_PATH", "/app/data/cleaned_imdb_data.csv")
        logging.basicConfig(
            filename='/app/logs/import_imdb_data.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

        def clean_text(text):
            if not text or isinstance(text, float):
                return ""
            try:
                return text.encode('utf-8', errors='replace').decode('utf-8')
            except Exception as e:
                logger.warning(f"Text cleaning error: {e}")
                return text.encode('ascii', errors='replace').decode('ascii')

        def create_movie_nodes(tx, movie_data):
            query = (
                "MERGE (m:Movie {imdb_id: $imdb_id}) "
                "SET m.title = $title, m.year = $year, m.genre = $genre, "
                "m.director = $director, m.rating = $rating, m.votes = $votes"
            )
            try:
                tx.run(query, **movie_data)
                logger.debug(f"Created/updated Neo4j node for {movie_data['title']}")
            except Exception as e:
                logger.error(f"Neo4j error for {movie_data['title']}: {e}")

        try:
            driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
            driver.verify_connectivity()
        except Exception as e:
            logger.error(f"Neo4j connection error: {e}")
            driver = None

        try:
            with transaction.atomic():
                Movie.objects.all().delete()
                logger.info("Cleared existing movies from database")
                with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
                    reader = csv.DictReader(csvfile)
                    movies_to_create = []
                    for row in reader:
                        try:
                            movie_data = {
                                'series_title': clean_text(row.get('series_title', '')),
                                'released_year': int(row['released_year']) if row.get('released_year') and row['released_year'].isdigit() else None,
                                'runtime': int(row['runtime'].replace(' min', '')) if row.get('runtime') else None,
                                'genre': clean_text(row.get('genre', '')),
                                'rating': float(row['rating']) if row.get('rating') else None,
                                'no_of_votes': int(row['no_of_votes']) if row.get('no_of_votes') else 0,
                                'director': clean_text(row.get('director', '')),
                                'star1': clean_text(row.get('star1', '')),
                                'star2': clean_text(row.get('star2', '')),
                                'star3': clean_text(row.get('star3', '')),
                                'star4': clean_text(row.get('star4', '')),
                                'overview': clean_text(row.get('overview', '')),
                                'poster_link': clean_text(row.get('poster_link', '')),
                                'awards': clean_text(row.get('awards', '')),
                                'certificate': clean_text(row.get('certificate', '')),
                                'gross': clean_text(row.get('gross', '')),
                                'meta_score': int(row['meta_score']) if row.get('meta_score') and row['meta_score'].isdigit() else None,
                            }
                            movies_to_create.append(Movie(**movie_data))
                            if driver:
                                with driver.session() as session:
                                    neo4j_data = {
                                        'imdb_id': clean_text(row.get('imdb_id', '')),
                                        'title': movie_data['series_title'],
                                        'year': movie_data['released_year'],
                                        'genre': movie_data['genre'],
                                        'director': movie_data['director'],
                                        'rating': movie_data['rating'],
                                        'votes': movie_data['no_of_votes'],
                                    }
                                    session.write(create_movie_nodes, neo4j_data)
                        except Exception as e:
                            logger.error(f"Error processing row {row.get('series_title', 'unknown')}: {e}")
                            continue
                    Movie.objects.bulk_create(movies_to_create, batch_size=1000)
                    logger.info(f"Imported {len(movies_to_create)} movies into database")
        except Exception as e:
            logger.error(f"Import error: {e}")
            raise
        finally:
            if driver:
                driver.close()
        logger.info("IMDb data import completed")
