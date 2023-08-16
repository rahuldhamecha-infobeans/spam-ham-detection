# import streamlit as st
from flask import Flask, render_template, request, flash
import pandas as pd
import numpy as np
import pickle
import string
from nltk.corpus import stopwords
import nltk
nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
from application_list import application_list
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
##import spotipy
import os
#from spotipy.oauth2 import SpotifyClientCredentials
from spotify import *
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler

ps = PorterStemmer()
pp_model = load_model('popularity_predict.h5')

app = Flask(__name__)

from spotify import  *

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))
X = ""


@app.context_processor
def inject_global_navbar_brand():
    navbar_brand = "<span>Infobeans</span> POCs"
    return dict(navbar_brand=navbar_brand)


@app.route('/')
def home_page():
    return render_template('home.html',application_list=application_list)


@app.route('/spam-detection', methods=['POST', 'GET'])
def investor():
    navbar_brand = "Spam <span>Detection</span>"
    result_message = '';
    if request.method == 'POST':
        result = request.form.get('message')
        print(result)
        transformed_sms = transform_text(result)
        # 2. vectorize
        vector_input = tfidf.transform([transformed_sms])
        # 3. predict
        result = model.predict(vector_input)[0]
        # 4. Display
        if result == 1:
            result_message = "Spam"
        else:
            result_message = "Not Spam"

    result = result_message

    return render_template('spam-ham-detection.html', **locals())

@app.route("/popularity-based-recommendation", methods=["GET", "POST"])
def popularity_based_recommendation():
    if request.method == "POST":
         songname=request.form.get("song")
    else:   
         songname=""
    new_song_df1 = find_song(songname)
    print(new_song_df1)
    if new_song_df1 is None:
        return render_template('song-recommendation.html',message='invalid song ') 
    new_song_df1 = new_song_df1.copy()
    new_song_df = new_song_df1.drop(columns=['name'])
    print(new_song_df1)
    csv_path = os.path.join(app.root_path, 'static/csv', 'song_data.csv')
    songdata= pd.read_csv(csv_path)
    songdata.drop_duplicates(keep='first',inplace=True)
    X= songdata.drop(['song_name','song_popularity','energy'],axis=1)  #axis=1 is by columns
    y= songdata['song_popularity']
    scaler= MinMaxScaler()
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=0)
    scaler.fit(X_train)
    scaled_x_train= scaler.transform(X_train)
    scaled_x_test = scaler.transform(X_test)

    #print(new_song_df)
    # Load the saved MinMaxScaler

    # Scale the new song features
    scaled_new_song = scaler.transform(new_song_df)
   # print(scaled_new_song.shape)
    # Check if the number of features matches the model's input shape
    if scaled_new_song.shape[1] == pp_model.input_shape[1]:
        # Make the prediction using the trained model
        predicted_popularity_scaled = pp_model.predict(scaled_new_song)
        

        # Inverse transform the scaled prediction to get the actual popularity value
        predicted_popularity = predicted_popularity_scaled[0][0] * 100
        
        print("Predicted Song Popularity:", predicted_popularity)
    else:
        print("Number of features in the new song data doesn't match the model's input shape.")
    # Predict popularity for the new song
    # Calculate the absolute difference between the predicted popularity of the new song and all popularity scores
    popularity_diffs = np.abs(predicted_popularity_scaled - pp_model.predict(scaled_x_train))

    # Sum the popularity differences along the axis 1 to get a single difference value per song
    popularity_diffs_sum = popularity_diffs.sum(axis=1)

    # Get the indices of songs with the smallest popularity differences (most similar)
    similar_song_indices = np.argsort(popularity_diffs_sum)[:10]  # Get top 10 similar songs

    # Get the names and predicted popularity scores of similar songs
    similar_song_names = songdata.iloc[similar_song_indices]['song_name']
    similar_song_popularities = predicted_popularity_scaled  # Use the same predicted popularity for all similar songs

    # Print similar song names and predicted popularities
    for song_name in similar_song_names:
        print("Song:", song_name)
        print("Predicted Popularity:", similar_song_popularities[0][0]*100)
        print("=" * 30)
    return render_template('song-recommendation.html', similar_songs=similar_song_names,song_predicted_rank=predicted_popularity,songname=songname)



if __name__ == '__main__':
    app.run(debug=True)
