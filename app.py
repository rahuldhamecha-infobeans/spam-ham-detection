# import streamlit as st
import os
from flask import Flask, render_template, request, flash
import numpy as np
import cv2
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from application_list import application_list
from keras.models import load_model
from keras.utils import load_img, img_to_array
import base64

ps = PorterStemmer()
app = Flask(__name__)

classification_model = load_model('image_classification.h5')
multiclass_classification_model = load_model('multi_image_classification.h5')

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
    return render_template('home.html', application_list=application_list)


@app.route('/spam-detection', methods=['POST', 'GET'])
def investor():
    navbar_brand = "Spam <span>Detection</span>"
    result_message = ''
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

        return render_template('multi-image-classification.html', label=label, filename=("pictures/"+filename))

    return render_template('multi-image-classification.html')

if __name__ == '__main__':
    app.run(debug=True)
