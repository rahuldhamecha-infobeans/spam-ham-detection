import os
import numpy as np
import cv2
import base64
from flask import render_template, Blueprint, request
from keras.models import load_model
from keras.utils import load_img, img_to_array


classification_model = load_model(os.path.join(os.path.dirname(
    __file__), 'models', 'image_classification.h5'))
multiclass_classification_model = load_model(os.path.join(os.path.dirname(
    __file__), 'models', 'multi_image_classification.h5'))

image_classification_blueprint = Blueprint(
    'imageclassification', __name__, template_folder='templates/')


@image_classification_blueprint.route('/single', methods=['GET', 'POST'])
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


@image_classification_blueprint.route('/multi', methods=['GET', 'POST'])
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
