from django.urls import path
from . import views

app_name = 'movies'

urlpatterns = [
    path('', views.home, name='home'),
    path('movie/<int:movie_id>/', views.movie_detail, name='movie_detail'),
    path('export-search-data/', views.export_search_data, name='export_search_data'),
    path('search-analytics/', views.search_analytics, name='search_analytics'),
]