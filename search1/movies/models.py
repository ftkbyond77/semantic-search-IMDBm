from django.db import models

class Movie(models.Model):
    series_title = models.CharField(max_length=255)
    released_year = models.IntegerField()
    runtime = models.CharField(max_length=50)
    genre = models.CharField(max_length=100)
    rating = models.FloatField()
    overview = models.TextField()
    director = models.CharField(max_length=100)
    star1 = models.CharField(max_length=100)
    star2 = models.CharField(max_length=100)

    def __str__(self):
        return self.series_title