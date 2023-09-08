from ib_tool import app
from flask import render_template
from application_list import application_list
from poc.songsdetection.views import songs_blueprint
from poc.spamham.views import spamham_blueprint
from poc.objectdetection.views import object_detection_blueprint
from poc.imageclassification.views import image_classification_blueprint
from poc.emailvalidator.views import email_validator_blueprint

@app.route('/')
def home_page():
    return render_template('home.html', application_list=application_list)


app.register_blueprint(spamham_blueprint,
                       url_prefix='/spam-ham')
app.register_blueprint(object_detection_blueprint,
                       url_prefix='/object-detection')
app.register_blueprint(songs_blueprint,
                       url_prefix='/songs')
app.register_blueprint(image_classification_blueprint,
                       url_prefix='/image-classification')
app.register_blueprint(email_validator_blueprint,
                       url_prefix='/email-validator')

if __name__ == '__main__':
    app.run(debug=True)
