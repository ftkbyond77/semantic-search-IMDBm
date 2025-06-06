# imdb_project/urls.py
from django.contrib import admin
from django.urls import path
from movies.views import home

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', home, name='home'),  # Root URL for home page
]