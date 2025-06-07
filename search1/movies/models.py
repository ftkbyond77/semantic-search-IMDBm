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
    rating = models.IntegerField(choices=[(i, i) for i in range(1, 6)])  # 1-5 stars
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('user', 'movie')

    def __str__(self):
        return f"{self.user.username} rated {self.movie.series_title} {self.rating} stars"