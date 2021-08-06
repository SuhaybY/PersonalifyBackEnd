import os
import json
from django.http import JsonResponse
from django.http.response import StreamingHttpResponse
from operator import itemgetter
import csv
import traceback


# Import model
# https://towardsdatascience.com/using-joblib-to-speed-up-your-python-pipelines-dd97440c653d
# import joblib

# model = joblib.load('modelPipe.pkl')

# Create user if DNE
def userLogin(req):
  if req.method == 'POST':
    try:
      req_params = json.loads(req.body)
      user = req_params['username']
      if not os.path.exists(f'user_data/{user}'):
        os.makedirs(f'user_data/{user}/history') 
      
      return JsonResponse({'Username': user})
    except Exception as e: 
      print(e)
      traceback.print_exc()
      return JsonResponse({'message': 'An error has occurred!'})

def getPlaylist(req):
  from .model.primary import main as ModelRun
  
  if req.method == 'POST':
    try:
      req_params = json.loads(req.body)
      user, playlist_url, num_songs = itemgetter('username', 'playlist_url', 'song_count')(req_params)
      num_songs = int(num_songs)
      req_id, req_date, results = ModelRun(user, playlist_url, num_songs)

      return JsonResponse({'results': results, 'id': req_id, 'date': req_date})
    except Exception as e: 
      print(e)
      traceback.print_exc()
      return JsonResponse({'message': 'An error has occurred!'})

def getHistory(req, id = ''):
  if req.method == 'POST':
    try:
      req_params = json.loads(req.body)
      user = req_params['username']
      h_list = os.listdir(f'user_data/{user}/history')
      if id == '':
        # Return entire history list
        h_list = [{'id':h.split('_')[0], 'date':h.split('_')[1].split('.')[0]} for h in h_list]
        return JsonResponse({'history': h_list})
      
      # Return specific
      for h in h_list:
        if h.split('_')[0] == id:
          results = []

          with open(f'user_data/{user}/history/{h}', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            results = [row for row in reader]
          return JsonResponse({id: results})
      
      return JsonResponse({'message': 'History item not found!'})
    except Exception as e: 
      print(e)
      traceback.print_exc()
      return JsonResponse({'message': 'An error has occurred!'})

def rateSongs(req):
  from .model.primary import dynamic_feedback_start as dynamicRun

  if req.method == 'POST':
    try:
      req_params = json.loads(req.body)
      user, url, recommendation_list, ratings = itemgetter('username', 'url', 'recommendation_list', 'ratings')(req_params)

      req_id, req_date, results = dynamicRun(user, url, recommendation_list, ratings)
      return JsonResponse({'results': results, 'id': req_id, 'date': req_date})
    except Exception as e: 
      print(e)
      traceback.print_exc()
      return JsonResponse({'message': 'An error has occurred!'})

def getStatus(req, id):
  if req.method == 'GET':
    try:
      # status = requests[id]
      return JsonResponse({'status': 'Testing...'})
    except Exception as e: 
      print(e)
      traceback.print_exc()
      return JsonResponse({'message': 'An error has occurred!'})