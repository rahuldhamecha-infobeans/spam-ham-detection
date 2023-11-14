import os
import numpy as np
import cv2
import base64
from flask import render_template, Blueprint, request
from keras.models import load_model

classification_model = load_model(os.path.join(os.path.dirname(
    __file__), 'models', 'face_mask_detection_final_30.h5'))

mask_classification_blueprint = Blueprint(
    'maskdetection', __name__, template_folder='templates/')

@mask_classification_blueprint.route('/single', methods=['GET', 'POST'])
def mask_classification():
    if request.method == "POST":
        file = request.files["imgfile"]

        # test_img = cv2.imdecode(np.frombuffer(
        #     file.read(), np.uint8), cv2.COLOR_BGR2RGB)
        # Decode the uploaded image
        test_img = cv2.imdecode(np.frombuffer(
            file.read(), np.uint8), cv2.IMREAD_COLOR)

        # Ensure the correct color channels (BGR to RGB)
        test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
        test_img = cv2.resize(test_img, (128, 128))
        # test_input = test_img.reshape((1, 128, 128, 3))
        test_input = test_img / 255.0
        test_input = np.expand_dims(test_input, axis=0)

        predicted_probabilities = classification_model.predict(test_input)
        predicted_labels = np.argmax(predicted_probabilities, axis=1)

        if predicted_labels == 0:
            label = "With Mask"
        elif predicted_labels == 1:
            label = "Without Mask"

        _, buffer = cv2.imencode(".jpg", test_img)
        base64_image = base64.b64encode(buffer).decode("utf-8")

        return render_template('mask-classification.html', label=label, base64_image=base64_image)

    return render_template('mask-classification.html')
