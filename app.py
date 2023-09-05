# import streamlit as st
import os
from flask import Flask, render_template, request, flash, Response, jsonify, request
from flask_restful import Resource, Api
import numpy as np
import cv2
import gunicorn
from camera import *
import pandas as pd
import pickle
import string
from nltk.corpus import stopwords
import nltk
import pygal
import numpy
import spacy
import requests
# Load the spaCy English model
nlp = spacy.load("en_core_web_sm")

nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
from application_list import application_list
from keras.models import load_model
from keras.utils import load_img, img_to_array
import base64
from sklearn.model_selection import train_test_split
from spotify import *
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler
import joblib
import os
ps = PorterStemmer()
pp_model = load_model('popularity_predict.h5')

app = Flask(__name__)

classification_model = load_model('image_classification.h5')
multiclass_classification_model = load_model('multi_image_classification.h5')
from spotify import *

API_URL = "https://api-inference.huggingface.co/models/atharvamundada99/bert-large-question-answering-finetuned-legal"
headers = {"Authorization": "Bearer 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"}

headings = ("Name", "Album", "Artist")
df1 = music_rec()
df1 = df1.head(15)


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
    navbar_brand = "<span>Infobeans</span> AI POCs"
    return dict(navbar_brand=navbar_brand)


@app.route('/')
def home_page():
    return render_template('home.html', application_list=application_list)


@app.route('/spam-detection', methods=['POST', 'GET'])
def investor():
    navbar_brand = "Spam <span>Detection</span>"
    result_message = ''
    result = ''

    if request.method == 'POST':
        result = request.form.get('message')
        transformed_sms = transform_text(result)
        # 2. vectorize
        vector_input = tfidf.transform([transformed_sms])
        # 3. predict
        prediction = model.predict(vector_input)[0]
        # 4. Display
        if prediction == 1:
            result_message = "Spam"
        else:
            result_message = "Not Spam"

    template_args = {
        'navbar_brand': navbar_brand,
        'message': result,
        'result': result_message,
    }

    return render_template('spam-ham-detection.html', **template_args)


@app.route("/popularity-based-recommendation", methods=["GET", "POST"])
def popularity_based_recommendation():
    title_brand = "Song <span>Recommendations</span>"
    if request.method == "POST":
        songname = request.form.get("song")
    else:
        songname = ""
    if songname == "":
        return render_template('song-recommendation.html', message='', title_brand=title_brand, )
    new_song_df1 = find_song(songname)
    if new_song_df1 is None:
        return render_template('song-recommendation.html', message='invalid song ', title_brand=title_brand, )
    new_song_df1 = new_song_df1.copy()
    new_song_df = new_song_df1.drop(columns=['name'])
    csv_path = os.path.join(app.root_path, 'static/csv', 'song_data.csv')
    songdata = pd.read_csv(csv_path)
    songdata.drop_duplicates(keep='first', inplace=True)
    X = songdata.drop(['song_name', 'song_popularity', 'energy'], axis=1)  # axis=1 is by columns
    y = songdata['song_popularity']
    scaler = MinMaxScaler()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    scaler.fit(X_train)
    scaled_x_train = scaler.transform(X_train)
    scaled_x_train = scaler.transform(X_train)
    # Scale the new song features
    scaled_new_song = scaler.transform(new_song_df)
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
    return render_template('song-recommendation.html', title_brand=title_brand, similar_songs=similar_song_names,
                           song_predicted_rank=predicted_popularity, songname=songname)


@app.route("/mood-based-recommendation", methods=["GET", "POST"])
def mood_based_recommendation():
    title_brand = "Song <span>Recommendations</span>"
    if request.method == "POST":
        songname = request.form.get("song")
    else:
        return render_template('mood-based-recommendation.html', common_mood='', title_brand=title_brand, )
    new_song_df = find_mood_based_song(songname)
    if new_song_df is None:
        return render_template('song-recommendation.html', message='invalid song ', title_brand=title_brand, )
    mood_csv_path = os.path.join(app.root_path, 'static/csv', 'data_moods.csv')
    df = pd.read_csv(mood_csv_path)
    X = df.loc[:, 'popularity':'time_signature']
    X['length'] = X['length'] / max(X['length'])
    model_dir = 'trained_model_classifiers'

    loaded_models = []
    loaded_models.append(
        ('Random Forest Classifier', joblib.load(os.path.join(model_dir, 'Random Forest Classifier.joblib'))))
    loaded_models.append(
        ('Gradient Boosting Classifier', joblib.load(os.path.join(model_dir, 'Gradient Boosting Classifier.joblib'))))
    loaded_models.append(('XGB Classifier', joblib.load(os.path.join(model_dir, 'XGB Classifier.joblib'))))
    loaded_models.append(
        ('Decision Tree Classifier', joblib.load(os.path.join(model_dir, 'Decision Tree Classifier.joblib'))))
    loaded_models.append(('LGBM Classifier', joblib.load(os.path.join(model_dir, 'LGBM Classifier.joblib'))))
    loaded_models.append(
        ('Support Vector Classifier', joblib.load(os.path.join(model_dir, 'Support Vector Classifier.joblib'))))
    loaded_models.append(('KNN Classifier', joblib.load(os.path.join(model_dir, 'KNN Classifier.joblib'))))
    # Scale the new song's data
    new_song_scaled = new_song_df.copy()
    new_song_scaled['length'] = new_song_scaled['length'] / max(X['length'])  # Scale the length feature
    new_song_scaled = new_song_scaled.drop(columns=['name'])
    mood_count = {}  # Dictionary to count predicted moods
    target_names = ['Happy', 'Sad', 'Energetic', 'Calm']
    # Predict mood using trained models
    for name, model in loaded_models:
        mood_prediction = model.predict(new_song_scaled)
        predicted_mood = target_names[mood_prediction[0]]  # Get the predicted mood label
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


@app.route("/facial-emotion-based-recommendation", methods=["GET", "POST"])
def facial_emotion_based_recommendation():
    title_brand = "Song <span>Recommendations</span>"
    # print(df1.to_json(orient='records'))
    return render_template('emotion-based-song-recommendation.html', title_brand=title_brand, headings=headings,
                           data=df1)


def gen(camera):
    while True:
        global df1
        frame, new_df1 = camera.get_frame()
        if frame is not None:
            df1 = new_df1  # Update the df1 with the new DataFrame
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/t')
def gen_table():
    return df1.to_json(orient='records')


@app.route('/image-classification', methods=['GET', 'POST'])
def image_classification():
    if request.method == "POST":
        file = request.files["imgfile"]

        test_img = cv2.imdecode(np.frombuffer(
            file.read(), np.uint8), cv2.IMREAD_COLOR)
        test_img = cv2.resize(test_img, (128, 128))
        test_input = test_img.reshape((1, 128, 128, 3))

        predicted_probabilities = classification_model.predict(test_input)
        predicted_labels = np.argmax(predicted_probabilities, axis=1)

        if predicted_labels == 0:
            label = "Cat"
        elif predicted_labels == 1:
            label = "Dog"

        _, buffer = cv2.imencode(".jpg", test_img)
        base64_image = base64.b64encode(buffer).decode("utf-8")

        return render_template('image-classification.html', label=label, base64_image=base64_image)

    return render_template('image-classification.html')


@app.route('/multi-image-classification', methods=['GET', 'POST'])
def multi_image_classification():
    if request.method == "POST":
        file = request.files["imgfile"]
        filename, extension = os.path.splitext(file.filename)
        filename = "classification" + extension
        upload_dir = os.path.join(app.root_path, "static/pictures")
        file.save(os.path.join(upload_dir, filename))

        image_path = upload_dir + "/" + filename
        new_height, new_width = 150, 150  # Replace with your desired dimensions

        # Create a custom generator for a single image
        def single_image_generator(image_path, target_size):
            img = load_img(image_path, target_size=target_size)
            img_array = img_to_array(img)
            img_array = img_array.reshape(
                (1,) + img_array.shape
            )  # Add a batch dimension
            img_array /= 255.0  # Rescale pixel values to [0, 1]
            return img_array

        # Generate the preprocessed image using the custom generator
        data_generator = single_image_generator(
            image_path, target_size=(new_height, new_width)
        )

        predict = multiclass_classification_model.predict(data_generator)
        predicted_max = np.argmax(predict, axis=-1)

        if predicted_max == 0:
            label = "Car"
        elif predicted_max == 1:
            label = "Cat"
        else:
            label = "Dog"

        return render_template('multi-image-classification.html', label=label, filename=("pictures/" + filename))

    return render_template('multi-image-classification.html')


@app.route('/object-detection')
def object_detection():
    navbar_brand = "<span>Live </span> Object Detection"
    return render_template('object-detection.html', navbar_brand=navbar_brand)


@app.route('/video-feed-for-object')
def video_feed_for_object():
    return Response(generate_webcam(VideoCameraForObject()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def generate_webcam(camera):
    while True:
        frame = camera.get_frame()
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


api = Api(app)


class SpamHamDetection(Resource):
    def get(self):
        response_data = {
            'percentage': 'GET',
            'suggestions': [
                'Test 1', 'Test 2', 'Test 3', 'Test 4'
            ]
        }
        return jsonify(response_data)

    def post(self):
        try:
        # Get the JSON data from the POST request
            data = request.get_json()

            # Check if the 'email_content' field exists in the JSON data
            if 'emailto_content' in data:
                email_content = data['emailto_content']
                email_reply = data['email_reply']
                detected_questions = preprocess_email_content(email_content)
                # Return the detected questions as JSON response
                total_qsn=len(detected_questions)
                unanswered_qsn=[]
                answered_qsn=[]
                is_ans=0
                if detected_questions:
                    for question in detected_questions:
                        output = query({
                            "inputs": {
                                "question": question,
                                "context": email_reply
                            },
                        })
                        if output:
                            if output['score'] > 0.2:
                                is_ans = is_ans+1
                                answered_qsn.append(str(question)) 
                            else:    
                                unanswered_qsn.append(str(question)) 
                                
                correct_reply_accuracy=(is_ans/total_qsn)*100
                return jsonify({"detected_questions": detected_questions,"unanswered_qsn":unanswered_qsn, "answered_qsn":answered_qsn,"accuracy":correct_reply_accuracy})

            else:
                return jsonify({"error": "Missing 'email_content' field in the request data"}), 400
        except Exception as e:
            return jsonify({"error": str(e)}), 500


api.add_resource(SpamHamDetection, '/api/spam-ham-email-detection')


def preprocess_email_content(email_content):
    # Define common greetings and closing phrases to remove
    common_phrases = ["hi", "hello", "hey", "best regards", "regards", "thank you", "thanks", "yours sincerely", "dear"]

    # Convert email content to lowercase for case-insensitive matching
    email_lower = email_content.lower()

    # Remove common phrases
    for phrase in common_phrases:
        email_lower = email_lower.replace(phrase, "")

    # Process the email content using spaCy
    doc = nlp(email_lower)

    # Initialize a list to store detected questions
    questions = []

    # Identify sentences with question patterns and convert them to strings
    for sentence in doc.sents:
        if "?" in sentence.text or any(token.lower_ in ("who", "what", "when", "where", "why", "how", "can", "is", "are", "do", "did", "could") for token in sentence):
            questions.append(str(sentence.text))  # Convert to string

    return questions

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()

@app.route('/email-validator', methods=['POST', 'GET'])
def email_validator():
    navbar_brand = "Email <span> Validator</span>"
    template_args = {
        'navbar_brand': navbar_brand,
    }
    return render_template('email-qna-validator.html', **template_args)





if __name__ == '__main__':
    app.run(debug=True)
