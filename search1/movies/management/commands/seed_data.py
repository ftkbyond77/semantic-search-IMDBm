from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from movies.models import Movie, Rating
from django.utils import timezone
import random

class Command(BaseCommand):
    help = 'Seed test user and ratings for existing movies'

    def handle(self, *args, **kwargs):
        # Create or update test user
        user, created = User.objects.get_or_create(
            username='testuser',
            defaults={'email': 'testuser@example.com'}
        )
        if created or not user.check_password('password123'):
            user.set_password('password123')
            user.save()
            action = 'Created' if created else 'Updated'
            self.stdout.write(self.style.SUCCESS(f'{action} testuser with password: password123'))

        # Seed ratings for existing movies
        movies = Movie.objects.all()
        if not movies:
            self.stdout.write(self.style.WARNING('No movies found in the database. Please import movie data first.'))
            return

        for movie in movies:
            Rating.objects.get_or_create(
                user=user,
                movie=movie,
                defaults={'rating': random.randint(1, 5), 'timestamp': timezone.now()}
            )
        self.stdout.write(self.style.SUCCESS(f'Seeded ratings for testuser on {len(movies)} movies'))