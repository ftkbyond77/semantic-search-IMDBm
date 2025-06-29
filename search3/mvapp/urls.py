from django.urls import path
from mvapp import views

urlpatterns = [
    path('', views.index, name='index'),
    path('api/search/', views.SearchView.as_view(), name='search'),
]