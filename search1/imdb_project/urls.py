from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.shortcuts import redirect

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('movies.urls')),
    path('', lambda request: redirect('movies:home'), name='root_redirect'),
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)