from django.urls import path, re_path
from . import views

urlpatterns = [
    path('login', views.userLogin, name='userLogin'),
    path('inputplaylist', views.getPlaylist, name='getPlaylist'),
    path('history', views.getHistory, name='getHistory'),
    re_path(r'^history/(?P<id>\w+)$', views.getHistory, name='getHistory'),
    path('ratesong', views.rateSong, name='rateSong'),
]