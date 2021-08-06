import pandas as pd
import re
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

class Spotify():
  def __init__(self):
    self.sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id="f3f14386465b4ac1b96c23194a0aec4b",
                                                           client_secret="3a42f3cca6c1415a9000ad9a3f3d5c1a"))

  # returns a 2-tuple of the playlist ID and the playlist name from a playlist name, by searching and using the first result
  def get_playlist(self, name):
    return (lambda x : (x["id"], x["name"]))(self.sp.search(name,limit=1,type="playlist")["playlists"]["items"][0])

  # returns a playlist ID from a URL
  def get_playlist_id_from_url(url):
    return re.findall('playlist\\/(\\w+)', url)[0]

  # returns a list of 2-tuples each containing a track title and artist for each track in the given playlist
  def get_playlist_tracks(self, playlist_id):
    return [(item["track"]["artists"][0]["name"],item["track"]["name"]) for item in self.sp.playlist(playlist_id=playlist_id)["tracks"]["items"]]

  # returns a list of track IDs from a playlist
  def get_playlist_track_ids(self, playlist_id):
    return [item["track"]["id"] for item in self.sp.playlist(playlist_id=playlist_id)["tracks"]["items"]]

  # returns a list of popularities, each corresponding to a given track ID
  def get_track_info(self, attribs, track_ids):
    data = []
    for track_id in track_ids:
      items = []
      for attrib in attribs:
        nested_attrib = attrib.split('.')
        item = self.sp.track(track_id)
        for i in nested_attrib:
          item = item[i]
        items.append(item)
      data.append(items)
    
    return pd.DataFrame(data, columns=attribs)

  # returns a track ID from a given track title and artist
  def get_id_from_track(self, title, artist):
    query = "track:" + title + " " + "artist:" + artist
    response = self.sp.search(query,limit=1,type="track")
    return response["tracks"]["items"][0]["id"]

  # returns all the track IDs for a list of 2-tuples, each containing a track title and artist 
  def get_ids_from_tracks(self, track_list=[]):
    return [self.get_id_from_track(track[0], track[1]) for track in track_list]

  # returns a pandas DataFrame of all the attributes for a given list of track IDs
  def fetch_spotify_attributes_from_ids(self, track_ids=[]):
    res = self.sp.audio_features(track_ids)
    popularities = self.get_track_info(['popularity'], track_ids)['popularity']
    df = pd.DataFrame(res)
    df = df.drop(['id', 'duration_ms', 'time_signature', 'analysis_url', 'track_href', 'uri','type'], axis=1)
    df.insert(loc=0, column='popularity', value=popularities)
    return df
