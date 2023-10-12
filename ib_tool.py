from flask import Flask, render_template,send_from_directory
from flask_mail import Mail,Message
from app_config import mail_config
import os

BASE_DIR = os.path.dirname(__file__)
UPLOAD_FOLDER = os.path.join(BASE_DIR,'uploads/')
# ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
# Create Ecommerce App To Use in multiple files
def create_ecommerce_app():
    app = Flask(__name__)
    return app

app = create_ecommerce_app()
app.config['SECRET_KEY'] = 'infobeans_app_key'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

email_config = mail_config()

# configuration of mail
app.config['MAIL_SERVER'] = email_config['MAIL_SERVER']
app.config['MAIL_PORT'] = email_config['MAIL_PORT']
app.config['MAIL_USERNAME'] = email_config['MAIL_USERNAME']
app.config['MAIL_PASSWORD'] = email_config['MAIL_PASSWORD']
app.config['MAIL_USE_TLS'] = email_config['MAIL_USE_TLS']
app.config['MAIL_USE_SSL'] = email_config['MAIL_USE_SSL']

def create_mail():
    mail = Mail(app)
    return mail

mail = create_mail()

@app.route('/uploads/<dir>/<name>')
def get_file_url(dir,name):
    path = os.path.join(app.config['UPLOAD_FOLDER'],dir)
    return send_from_directory(path, name)

@app.route('/uploads/<dir>/final-frames/<name>')
def get_final_frame(dir,name):
    path = os.path.join(app.config['UPLOAD_FOLDER'],dir,'final-frames')
    return send_from_directory(path, name)

app.add_url_rule("/uploads/<dir>/<name>", endpoint="get_file_url", build_only=True)
app.add_url_rule("/uploads/<dir>/final-frames/<name>", endpoint="get_final_frame", build_only=True)

import ib_aitool.register_application
import ib_aitool.context_processor
