from django.db import models
from django.contrib.auth.models import User

class Movie(models.Model):
    series_title = models.CharField(max_length=255)
    released_year = models.IntegerField(null=True, blank=True)
    runtime = models.CharField(max_length=50)
    genre = models.CharField(max_length=255)
    rating = models.FloatField()
    overview = models.TextField()
    director = models.CharField(max_length=255)
    star1 = models.CharField(max_length=255)
    star2 = models.CharField(max_length=255)
    embedding = models.JSONField(null=True, blank=True)

    def __str__(self):
        return self.series_title

class Rating(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    movie = models.ForeignKey(Movie, on_delete=models.CASCADE, related_name='user_ratings')
    rating = models.IntegerField(choices=[(i, i) for i in range(1, 6)])
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('user', 'movie')

    def __str__(self):
        return f"{self.user.username} rated {self.movie.series_title} {self.rating} stars"

class SearchHistory(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)  # Allow anonymous users
    query = models.CharField(max_length=255)
    search_type = models.CharField(max_length=20, choices=[('keyword', 'Keyword'), ('semantic', 'Semantic')])
    results_count = models.IntegerField(default=0)  # Number of results returned
    session_id = models.CharField(max_length=100, null=True, blank=True)  # Track anonymous users
    ip_address = models.GenericIPAddressField(null=True, blank=True)  # Track IP for analytics
    user_agent = models.TextField(null=True, blank=True)  # Browser info
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-timestamp']

    def __str__(self):
        user_info = self.user.username if self.user else f"Anonymous({self.session_id[:8]})"
        return f"{user_info} searched '{self.query}' ({self.search_type}) - {self.results_count} results"