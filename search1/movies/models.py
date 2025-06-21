from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone

class Movie(models.Model):
    series_title = models.CharField(max_length=255, unique=True)
    released_year = models.PositiveIntegerField(null=True, blank=True)
    runtime = models.CharField(max_length=50, blank=True)
    genre = models.CharField(max_length=255, blank=True)
    rating = models.FloatField(null=True, blank=True)
    no_of_votes = models.PositiveIntegerField(default=0)
    director = models.CharField(max_length=255, blank=True)
    star1 = models.CharField(max_length=255, blank=True)
    star2 = models.CharField(max_length=255, blank=True)
    star3 = models.CharField(max_length=255, blank=True)
    star4 = models.CharField(max_length=255, blank=True)
    overview = models.TextField(blank=True)
    poster_link = models.URLField(blank=True)
    awards = models.TextField(blank=True)
    certificate = models.CharField(max_length=50, blank=True)
    embedding = models.JSONField(null=True, blank=True)
    gross = models.CharField(max_length=100, blank=True)
    meta_score = models.PositiveIntegerField(null=True, blank=True)

    def __str__(self):
        return self.series_title

    class Meta:
        db_table = 'movies_movie'

class Rating(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    movie = models.ForeignKey(Movie, on_delete=models.CASCADE, related_name='user_ratings')
    rating = models.IntegerField(null=True, blank=True)
    timestamp = models.DateTimeField(default=timezone.now, null=True, blank=True)

    class Meta:
        unique_together = ('user', 'movie')
        indexes = [
            models.Index(fields=['user', 'movie']),
        ]

class SearchHistory(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    session_id = models.CharField(max_length=255, null=True, blank=True)
    query = models.CharField(max_length=255, null=True, blank=True)
    search_type = models.CharField(max_length=50, default='keyword', null=True, blank=True)
    results_count = models.IntegerField(default=0, null=True, blank=True)
    ip_address = models.CharField(max_length=45, null=True, blank=True)
    user_agent = models.TextField(null=True, blank=True)
    timestamp = models.DateTimeField(default=timezone.now, null=True, blank=True)

    def __str__(self):
        return f"{self.query or 'Unknown Query'} ({self.search_type or 'N/A'})"

    class Meta:
        indexes = [
            models.Index(fields=['user', 'timestamp']),
            models.Index(fields=['session_id', 'timestamp']),
        ]
