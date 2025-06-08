from django.urls import path
from . import views
from django.shortcuts import redirect

app_name = 'movies'

urlpatterns = [
    # Redirect root URL to login
    path('', lambda request: redirect('movies:login'), name='root_redirect'),

    # Authentication routes
    path('login/', views.login_view, name='login'),
    path('signup/', views.signup_view, name='signup'),
    path('logout/', views.logout_view, name='logout'),

    # Main app routes
    path('home/', views.home, name='home'),
    path('movie/<int:movie_id>/', views.movie_detail, name='movie_detail'),
    path('export-search-data/', views.export_search_data, name='export_search_data'),
    path('search-analytics/', views.search_analytics, name='search_analytics'),
]
