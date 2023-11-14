import os
import joblib
import pandas as pd
from flask import render_template, request, jsonify, render_template, Blueprint, Response
from poc.songsdetection.camera import *
from poc.songsdetection.spotify import *

headings = ("Name", "Album", "Artist")
df1 = music_rec()
df1 = df1.head(15)

songs_blueprint = Blueprint(
    'songsdetection', __name__, template_folder='templates/')


@songs_blueprint.route("/mood-based-recommendation", methods=["GET", "POST"])
def mood_based_recommendation():
    title_brand = "Song <span>Recommendations</span>"
    if request.method == "POST":
        songname = request.form.get("song")
    else:
        return render_template('mood-based-recommendation.html', common_mood='', title_brand=title_brand, )
    new_song_df = find_mood_based_song(songname)
    if new_song_df is None:
        return render_template('song-recommendation.html', message='invalid song ', title_brand=title_brand, )
    # mood_csv_path = os.path.join(app.root_path, 'static/csv', 'data_moods.csv')
    mood_csv_path = os.path.join(
        os.path.dirname(__file__), 'csv', 'data_moods.csv')

    df = pd.read_csv(mood_csv_path)
    X = df.loc[:, 'popularity':'time_signature']
    X['length'] = X['length'] / max(X['length'])
    model_dir = os.path.join(
        os.path.dirname(__file__), 'trained_model_classifiers')

    loaded_models = []
    loaded_models.append(
        ('Random Forest Classifier', joblib.load(os.path.join(model_dir, 'Random Forest Classifier.joblib'))))
    loaded_models.append(
        ('Gradient Boosting Classifier', joblib.load(os.path.join(model_dir, 'Gradient Boosting Classifier.joblib'))))
    loaded_models.append(('XGB Classifier', joblib.load(
        os.path.join(model_dir, 'XGB Classifier.joblib'))))
    loaded_models.append(
        ('Decision Tree Classifier', joblib.load(os.path.join(model_dir, 'Decision Tree Classifier.joblib'))))
    loaded_models.append(('LGBM Classifier', joblib.load(
        os.path.join(model_dir, 'LGBM Classifier.joblib'))))
    loaded_models.append(
        ('Support Vector Classifier', joblib.load(os.path.join(model_dir, 'Support Vector Classifier.joblib'))))
    loaded_models.append(('KNN Classifier', joblib.load(
        os.path.join(model_dir, 'KNN Classifier.joblib'))))
    # Scale the new song's data
    new_song_scaled = new_song_df.copy()
    new_song_scaled['length'] = new_song_scaled['length'] / \
        max(X['length'])  # Scale the length feature
    new_song_scaled = new_song_scaled.drop(columns=['name'])
    mood_count = {}  # Dictionary to count predicted moods
    target_names = ['Happy', 'Sad', 'Energetic', 'Calm']
    # Predict mood using trained models
    for name, model in loaded_models:
        mood_prediction = model.predict(new_song_scaled)
        # Get the predicted mood label
        predicted_mood = target_names[mood_prediction[0]]
        if predicted_mood in mood_count:
            mood_count[predicted_mood] += 1
        else:
            mood_count[predicted_mood] = 1

        # Find the most common predicted mood
        most_common_mood = max(mood_count, key=mood_count.get)

    # print(mood_count)
    # print('\033[1mPredicted Mood for ' + new_song_df['name'] + '\033[0m' + ': \033[1m' + most_common_mood + '\033[0m')
    similar_songs = []
    similar_songs_by_mood = df[df['mood'] == most_common_mood]
    similar_songs.append(similar_songs_by_mood)
    similar_songs_df = pd.concat(similar_songs, ignore_index=True)
    print("Similar Songs with Predicted Mood:")
    print(similar_songs_df[['name', 'mood']])
    return render_template('mood-based-recommendation.html', title_brand=title_brand, songname=songname,
                           similar_songs=similar_songs_df, common_mood=most_common_mood)


@songs_blueprint.route("/facial-emotion-based-recommendation", methods=["GET", "POST"])
def facial_emotion_based_recommendation():
    title_brand = "Song <span>Recommendations</span>"
    return render_template('emotion-based-song-recommendation.html', title_brand=title_brand, headings=headings, data=df1)


@songs_blueprint.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@songs_blueprint.route('/t')
def gen_table():
    return df1.to_json(orient='records')

def gen(camera):
    while True:
        global df1
        frame, new_df1 = camera.get_frame()
        if frame is not None:
            df1 = new_df1  # Update the df1 with the new DataFrame
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
