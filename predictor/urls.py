
from .views import *
from django.urls import path
urlpatterns = [
path('', predict, name='scan'),
path("results/<str:result_id>/", view_results, name="view_results"),
path("getRagRes/<int:result_id>/", get_initial_rag_response, name="get_initial_rag_response"),
path("submitRagQuery/", submit_rag_query, name="submit_rag_query"),
path("makepublic/<int:result_id>", make_public, name="makeScanPublic"),
path("makeprivate/<int:result_id>", make_private, name="makeScanPrivate"),
path("delete/<int:result_id>", delete_scan, name="deleteScan"),
]