
from .views import *
from django.urls import path
urlpatterns = [
path('login/', login_view, name='login'),
path('signup/', sign_up, name='signup'),
path('logout/', logout_user, name='logout'),
path('home/', home, name='home'),
]
