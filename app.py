# import streamlit as st
from flask import Flask, render_template, request, flash

import pickle
import string
from nltk.corpus import stopwords
import nltk
nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
from application_list import application_list

ps = PorterStemmer()
app = Flask(__name__)


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


if __name__ == '__main__':
    app.run(debug=True)
