
from .views import *
from django.urls import path
urlpatterns = [
path('', predict, name='scan'),
path("results/<int:result_id>/", view_results, name="view_results"),
path("getRagRes/<int:result_id>/", get_initial_rag_response, name="get_initial_rag_response"),
path("submitRagQuery/", submit_rag_query, name="submit_rag_query"),

]