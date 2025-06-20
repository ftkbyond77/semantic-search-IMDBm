from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone

class Movie(models.Model):
    series_title = models.CharField(max_length=255)
    released_year = models.IntegerField()
    runtime = models.CharField(max_length=50, blank=True)
    genre = models.CharField(max_length=255, blank=True)
    rating = models.FloatField(null=True, blank=True)
    no_of_votes = models.IntegerField(default=0)
    director = models.CharField(max_length=255, blank=True)
    star1 = models.CharField(max_length=255, blank=True)
    star2 = models.CharField(max_length=255, blank=True)
    star3 = models.CharField(max_length=255, blank=True)
    star4 = models.CharField(max_length=255, blank=True)
    overview = models.TextField(blank=True)
    poster_link = models.URLField(blank=True)
    awards = models.CharField(max_length=255, blank=True)

    def __str__(self):
        return self.series_title

    class Meta:
        indexes = [
            models.Index(fields=['series_title']),
            models.Index(fields=['genre']),
            models.Index(fields=['director']),
        ]

class Rating(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    movie = models.ForeignKey(Movie, on_delete=models.CASCADE, related_name='user_ratings')
    rating = models.IntegerField()
    timestamp = models.DateTimeField(default=timezone.now)

    class Meta:
        unique_together = ('user', 'movie')
        indexes = [
            models.Index(fields=['user', 'movie']),
        ]

class SearchHistory(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    session_id = models.CharField(max_length=255, blank=True)
    query = models.CharField(max_length=255)
    search_type = models.CharField(max_length=50, default='keyword')
    results_count = models.IntegerField(default=0)
    ip_address = models.CharField(max_length=45, blank=True)
    user_agent = models.TextField(blank=True)
    timestamp = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return f"{self.query} ({self.search_type})"

    class Meta:
        indexes = [
            models.Index(fields=['user', 'timestamp']),
            models.Index(fields=['session_id', 'timestamp']),
        ]