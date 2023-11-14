import pandas as pd
import numpy as np
import spotipy
import os
from spotipy.oauth2 import SpotifyClientCredentials
from collections import defaultdict
import Levenshtein
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id="357958d690d54634b3f3dd218988bf83",
                                                           client_secret="57947d24dfdd418698691ec29c52c247"))
def find_song(name):
    song_data = defaultdict()
    results = sp.search(q='track: {}'.format(name), limit=1)

    if results['tracks']['items'] == []:
        return None

    result_track_name = results['tracks']['items'][0]['name']
    
    # Calculate Levenshtein distance
    distance = Levenshtein.distance(name.lower(), result_track_name.lower())
    
    # Set a threshold for similarity
    similarity_threshold = 100  # Adjust as needed
    
    if distance > similarity_threshold:
        return None

    results = results['tracks']['items'][0]
    track_id = results['id']
    audio_features = sp.audio_features(track_id)[0]
    
    song_data['name'] = [name]
    song_data['song_duration_ms'] = [results['duration_ms']]
    song_data['acousticness'] = [audio_features['acousticness']]
    song_data['danceability'] = [audio_features['danceability']]
    song_data['instrumentalness'] = [audio_features['instrumentalness']]
    #song_data['energy'] = [audio_features['energy']]
    song_data['key'] = [audio_features['key']]
    song_data['liveness'] = [audio_features['liveness']]
    song_data['loudness'] = [audio_features['loudness']]
    song_data['audio_mode'] = [audio_features['mode']]
    song_data['speechiness'] = [audio_features['speechiness']]
    song_data['tempo'] = [audio_features['tempo']]
    song_data['time_signature'] = [audio_features['time_signature']]
    song_data['audio_valence'] = [audio_features['valence']]
    #print(audio_features)
    
    #for key, value in audio_features.items():
     #    song_data[key] = value

    return pd.DataFrame(song_data)


def find_mood_based_song(name):
    song_data = defaultdict()
    results = sp.search(q='track: {}'.format(name), limit=1)

    if results['tracks']['items'] == []:
        return None

    result_track_name = results['tracks']['items'][0]['name']
    
    # Calculate Levenshtein distance
    distance = Levenshtein.distance(name.lower(), result_track_name.lower())
    
    # Set a threshold for similarity
    similarity_threshold = 100  # Adjust as needed
    
    if distance > similarity_threshold:
        return None

    results = results['tracks']['items'][0]
    track_id = results['id']
    audio_features = sp.audio_features(track_id)[0]

    song_data['name'] = [name]
    song_data['popularity'] = [results['popularity']]
    song_data['length'] = [results['duration_ms']]
    song_data['danceability'] = [audio_features['danceability']]
    song_data['acousticness'] = [audio_features['acousticness']]
    song_data['energy'] = [audio_features['energy']]
    song_data['instrumentalness'] = [audio_features['instrumentalness']]
    song_data['liveness'] = [audio_features['liveness']]
    song_data['valence'] = [audio_features['valence']]
    song_data['loudness'] = [audio_features['loudness']]
    song_data['speechiness'] = [audio_features['speechiness']]
    song_data['tempo'] = [audio_features['tempo']]
    song_data['key'] = [audio_features['key']]    
    song_data['time_signature'] = [audio_features['time_signature']]
    
    #print(audio_features)
    
    #for key, value in audio_features.items():
     #    song_data[key] = value

    return pd.DataFrame(song_data)