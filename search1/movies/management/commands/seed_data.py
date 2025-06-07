from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from movies.models import Movie, Rating
import random

class Command(BaseCommand):
    help = 'Seed movies and ratings for testing'

    def handle(self, *args, **kwargs):
        # Create or get a user
        user, created = User.objects.get_or_create(
            username='testuser',
            defaults={'password': 'password123'}  # Note: Use set_password for real apps
        )
        if created:
            user.set_password('password123')
            user.save()
            self.stdout.write(self.style.SUCCESS('Created testuser'))

        # Seed movies if none exist
        if not Movie.objects.exists():
            movies = [
                {'series_title': 'The Shawshank Redemption', 'released_year': 1994, 'runtime': '2h 22m', 'genre': 'Drama', 'rating': 9.3, 'overview': 'Two men in prison', 'director': 'Frank Darabont', 'star1': 'Tim Robbins', 'star2': 'Morgan Freeman'},
                {'series_title': 'The Godfather', 'released_year': 1972, 'runtime': '2h 55m', 'genre': 'Crime, Drama', 'rating': 9.2, 'overview': 'Mafia family saga', 'director': 'Francis Ford Coppola', 'star1': 'Marlon Brando', 'star2': 'Al Pacino'},
                {'series_title': 'Inception', 'released_year': 2010, 'runtime': '2h 28m', 'genre': 'Sci nemzet-Fi, Action', 'rating': 8.8, 'overview': 'Dream within a dream', 'director': 'Christopher Nolan', 'star1': 'Leonardo DiCaprio', 'star2': 'Joseph Gordon-Levitt'},
                {'series_title': 'Pulp Fiction', 'released_year': 1994, 'runtime': '2h 34m', 'genre': 'Crime, Drama', 'rating': 8.9, 'overview': 'Interwoven crime stories', 'director': 'Quentin Tarantino', 'star1': 'John Travolta', 'star2': 'Samuel L. Jackson'},
                {'series_title': 'The Dark Knight', 'released_year': 2008, 'runtime': '2h 32m', 'genre': 'Action, Crime', 'rating': 9.0, 'overview': 'Batman vs. Joker', 'director': 'Christopher Nolan', 'star1': 'Christian Bale', 'star2': 'Heath Ledger'},
            ]
            for movie_data in movies:
                Movie.objects.create(
                    series_title=movie_data['series_title'],
                    released_year=movie_data['released_year'],
                    runtime=movie_data['runtime'],
                    genre=movie_data['genre'],
                    rating=movie_data['rating'],
                    overview=movie_data['overview'],
                    director=movie_data['director'],
                    star1=movie_data['star1'],
                    star2=movie_data['star2'],
                    embedding=None
                )
            self.stdout.write(self.style.SUCCESS('Seeded 5 movies'))

        # Seed ratings
        movies = Movie.objects.all()
        for movie in movies:
            Rating.objects.get_or_create(
                user=user,
                movie=movie,
                defaults={'rating': random.randint(1, 5)}
            )
        self.stdout.write(self.style.SUCCESS('Seeded ratings for testuser'))