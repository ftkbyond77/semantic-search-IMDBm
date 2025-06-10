from django.db import models
from django.contrib.auth.models import User

class Movie(models.Model):
    series_title = models.CharField(max_length=255)
    released_year = models.IntegerField(null=True, blank=True)
    certificate = models.CharField(max_length=50, null=True, blank=True)
    runtime = models.CharField(max_length=50)
    genre = models.CharField(max_length=255)
    rating = models.FloatField()
    overview = models.TextField()
    meta_score = models.FloatField(null=True, blank=True)
    director = models.CharField(max_length=255)
    star1 = models.CharField(max_length=255)
    star2 = models.CharField(max_length=255)
    star3 = models.CharField(max_length=255, blank=True)
    star4 = models.CharField(max_length=255, blank=True)
    no_of_votes = models.IntegerField(default=0)
    gross = models.FloatField(null=True, blank=True)
    poster_link = models.URLField(max_length=500, blank=True)
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
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    query = models.CharField(max_length=255)
    search_type = models.CharField(max_length=20, choices=[('keyword', 'Keyword'), ('semantic', 'Semantic')])
    results_count = models.IntegerField(default=0)
    session_id = models.CharField(max_length=100, null=True, blank=True)
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    user_agent = models.TextField(null=True, blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-timestamp']

    def __str__(self):
        user_info = self.user.username if self.user else f"Anonymous({self.session_id[:8]})"
        return f"{user_info} searched '{self.query}' ({self.search_type}) - {self.results_count} results"