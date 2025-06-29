from django.db import models
import numpy as np

class Movie(models.Model):
    series_title = models.CharField(max_length=255)
    released_year = models.CharField(max_length=4)
    certificate = models.CharField(max_length=10, blank=True)
    runtime = models.CharField(max_length=20, blank=True)
    genre = models.CharField(max_length=100)
    imdb_rating = models.FloatField()
    overview = models.TextField()
    meta_score = models.FloatField(null=True, blank=True)
    director = models.CharField(max_length=100)
    star1 = models.CharField(max_length=100)
    star2 = models.CharField(max_length=100)
    star3 = models.CharField(max_length=100)
    star4 = models.CharField(max_length=100)
    no_of_votes = models.IntegerField()
    gross = models.BigIntegerField(null=True, blank=True)
    embedding = models.JSONField(null=True, blank=True)

    def __str__(self):
        return self.series_title