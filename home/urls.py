
from .views import *
from django.urls import path
urlpatterns = [
path('', index, name='index'),
path('about/', about, name='about'),
path('contactUs/', contact, name='contactUs'),
path('futureWork/', futureWork, name='futureWork'),

]