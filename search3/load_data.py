import pandas as pd
from django.conf import settings
from mvapp.models import Movie
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import os

def load_data():
    # Load CSV
    df = pd.read_csv('mvapp/data/cleaned_imdb_data.csv')
    
    # SQLite: Save to Django model
    for _, row in df.iterrows():
        Movie.objects.create(
            series_title=row['Series_Title'],
            released_year=row['Released_Year'],
            certificate=row['Certificate'],
            runtime=row['Runtime'],
            genre=row['Genre'],
            imdb_rating=row['IMDB_Rating'],
            overview=row['Overview'],
            meta_score=row['Meta_score'],
            director=row['Director'],
            star1=row['Star1'],
            star2=row['Star2'],
            star3=row['Star3'],
            star4=row['Star4'],
            no_of_votes=row['No_of_Votes'],
            gross=row['Gross']
        )
    
    # Semantic Search: Generate embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(df['Overview'].tolist())
    for i, movie in enumerate(Movie.objects.all()):
        movie.embedding = embeddings[i].tolist()
        movie.save()
    
    # Neo4j: Create nodes and relationships
    driver = GraphDatabase.driver("bolt://neo4j:7687", auth=("neo4j", "secretpassword"))
    with driver.session() as session:
        for _, row in df.iterrows():
            session.run(
                """
                MERGE (m:Movie {title: $title})
                MERGE (d:Director {name: $director})
                MERGE (s1:Actor {name: $star1})
                MERGE (s2:Actor {name: $star2})
                MERGE (s3:Actor {name: $star3})
                MERGE (s4:Actor {name: $star4})
                MERGE (m)-[:DIRECTED_BY]->(d)
                MERGE (m)-[:STARRING]->(s1)
                MERGE (m)-[:STARRING]->(s2)
                MERGE (m)-[:STARRING]->(s3)
                MERGE (m)-[:STARRING]->(s4)
                """,
                title=row['Series_Title'], director=row['Director'],
                star1=row['Star1'], star2=row['Star2'],
                star3=row['Star3'], star4=row['Star4']
            )
    driver.close()

if __name__ == "__main__":
    load_data()