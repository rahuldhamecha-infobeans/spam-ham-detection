import os
import nltk
from nltk.corpus import stopwords
import requests
from nltk.stem.porter import PorterStemmer
from flask import render_template, request, jsonify, render_template, Blueprint
from flask_restful import Resource, Api
import pickle
import spacy
import string
nlp = spacy.load("en_core_web_sm")


tfidf = pickle.load(open(os.path.join(os.path.dirname(
    __file__), 'models', 'vectorizer.pkl'), 'rb'))
model = pickle.load(open(os.path.join(os.path.dirname(
    __file__), 'models', 'spamham.pkl'), 'rb'))

ps = PorterStemmer()

API_URL = "https://api-inference.huggingface.co/models/atharvamundada99/bert-large-question-answering-finetuned-legal"
headers = {"Authorization": "Bearer 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"}

spamham_blueprint = Blueprint(
    'spamham', __name__, template_folder='templates/')

api = Api(spamham_blueprint)


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
                total_qsn = len(detected_questions)
                unanswered_qsn = []
                answered_qsn = []
                is_ans = 0
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

                correct_reply_accuracy = (is_ans/total_qsn)*100
                return jsonify({"detected_questions": detected_questions, "unanswered_qsn": unanswered_qsn, "answered_qsn": answered_qsn, "accuracy": correct_reply_accuracy})

            else:
                return jsonify({"error": "Missing 'email_content' field in the request data"}), 400
        except Exception as e:
            return jsonify({"error": str(e)}), 500


def preprocess_email_content(email_content):
    # Define common greetings and closing phrases to remove
    common_phrases = ["hi", "hello", "hey", "best regards",
                      "regards", "thank you", "thanks", "yours sincerely", "dear"]

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


@spamham_blueprint.route('/', methods=['POST', 'GET'])
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


api.add_resource(SpamHamDetection, '/api/spam-ham-email-detection')
