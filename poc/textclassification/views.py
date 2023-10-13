import os
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from flask import render_template, request, render_template, Blueprint
import pickle
import string

tfidf = pickle.load(open(os.path.join(os.path.dirname(
    __file__), 'models', 'vectorizer_new.pkl'), 'rb'))
model = pickle.load(open(os.path.join(os.path.dirname(
    __file__), 'models', 'model_new.pkl'), 'rb'))

ps = PorterStemmer()

textclassification_blueprint = Blueprint(
    'textclassification', __name__, template_folder='templates/')


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


@textclassification_blueprint.route('/', methods=['POST', 'GET'])
def investor():
    navbar_brand = "Text Classification by <span>Category</span>"
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
        result_message = prediction

    template_args = {
        'navbar_brand': navbar_brand,
        'message': result,
        'result': result_message,
    }

    return render_template('text-classificationbycategory.html', **template_args)
