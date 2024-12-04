

from django.urls import path

from app import views


urlpatterns = [
    path('', views.index, name='index'),
    path ('age_detection', views.age_detection, name='age_detection'),
    
]