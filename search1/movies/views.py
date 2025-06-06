from django.shortcuts import render

# Create your views here.
from django.shortcuts import render
from .models import Movie

def home(request):
    movies = Movie.objects.all()  # Fetch all movies from the database
    return render(request, 'movies/home.html', {'movies': movies})