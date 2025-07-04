from django.contrib import admin
from .models import Movie, Rating

@admin.register(Movie)
class MovieAdmin(admin.ModelAdmin):
    list_display = ('series_title', 'released_year', 'runtime', 'genre', 'rating', 'director', 'star1', 'star2')
    list_filter = ('genre', 'released_year', 'director')
    search_fields = ('series_title', 'genre', 'director', 'star1', 'star2', 'overview')
    ordering = ('-rating', 'series_title')
    readonly_fields = ('embedding',)

    def get_queryset(self, request):
        return super().get_queryset(request).order_by('-rating')

@admin.register(Rating)
class RatingAdmin(admin.ModelAdmin):
    list_display = ('user', 'movie', 'rating', 'timestamp')
    list_filter = ('rating', 'timestamp', 'user')
    search_fields = ('user__username', 'movie__series_title')
    ordering = ('-timestamp',)
    raw_id_fields = ('user', 'movie')  # Improves performance for large datasets

    def get_queryset(self, request):
        return super().get_queryset(request).select_related('user', 'movie')