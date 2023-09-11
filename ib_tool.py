from flask import Flask, render_template,send_from_directory
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

@app.route('/uploads/<dir>/<name>')
def get_file_url(dir,name):
    path = os.path.join(app.config['UPLOAD_FOLDER'],dir)
    return send_from_directory(path, name)

app.add_url_rule(
    "/uploads/<dir>/<name>", endpoint="get_file_url", build_only=True
)

import ib_aitool.register_application
import ib_aitool.context_processor
