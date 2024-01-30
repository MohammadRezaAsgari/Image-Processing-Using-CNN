from django.urls import path
from .views import ProcessApiView

urlpatterns = [
    path('process_image/', ProcessApiView.as_view()),
]
