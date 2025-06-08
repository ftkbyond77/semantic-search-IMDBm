from django.urls import path
from . import views

app_name = 'movies'

urlpatterns = [
    # Authentication routes
    path('login/', views.login_view, name='login'),
    path('signup/', views.signup_view, name='signup'),  
    path('logout/', views.logout_view, name='logout'),

    path('', views.home, name='home'),
    path('movie/<int:movie_id>/', views.movie_detail, name='movie_detail'),

    # Admin-only routes
    path('export-search-data/', views.export_search_data, name='export_search_data'),
    path('search-analytics/', views.search_analytics, name='search_analytics'),
]
