from django.urls import path
from .views import predict_mood

urlpatterns = [
    path('predict/', predict_mood),
]
