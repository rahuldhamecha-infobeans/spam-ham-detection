
import requests
from flask import request, jsonify, render_template, Blueprint,make_response
from flask_restful import Resource, Api
import spacy
import string
nlp = spacy.load("en_core_web_sm")

API_URL = "https://api-inference.huggingface.co/models/atharvamundada99/bert-large-question-answering-finetuned-legal"
headers = {"Authorization": "Bearer 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"}

email_validator_blueprint = Blueprint(
    'emailvalidator', __name__, template_folder='templates/')

api = Api(email_validator_blueprint)


@email_validator_blueprint.route('/', methods=['POST', 'GET'])
def email_validator():
    navbar_brand = "Email <span> Validator</span>"
    template_args = {
        'navbar_brand': navbar_brand,
    }
    return render_template('email-qna-validator.html', **template_args)


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
            unanswered_qsn = []
            answered_qsn = []
            # Check if the 'email_content' field exists in the JSON data
            if 'emailto_content' in data:
                email_content = data['emailto_content']
                email_reply = data['email_reply']
                email_reply = email_reply.replace('\n', "")
                detected_questions = preprocess_email_content(email_content)
                # Return the detected questions as JSON response
                total_qsn = len(detected_questions)
                is_ans = 0
                if len(detected_questions):
                    for question in detected_questions:
                        output = query({
                            "inputs": {
                                "question": question,
                                "context": email_reply
                            },
                        })
                        if output:
                            if output['score'] > 0.7:
                                is_ans = is_ans+1
                                answered_qsn.append(str(question))
                            else:
                                unanswered_qsn.append(str(question))

                correct_reply_accuracy = '{:.2f}'.format((is_ans/total_qsn)*100)
                response_data = jsonify({"detected_questions": detected_questions, "unanswered_qsn": unanswered_qsn, "answered_qsn": answered_qsn, "accuracy": correct_reply_accuracy})
                return make_response(response_data, 200)
            else:
                response_data = jsonify({"error": "Missing 'email_content' field in the request data"})
                return make_response(response_data, 400)
        except Exception as e:
            response_data = jsonify({"error": str(e)})
            return make_response(response_data,500)


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
    if response:
        return response.json()
    else:
        return {}


api.add_resource(SpamHamDetection, '/api/spam-ham-email-detection')
