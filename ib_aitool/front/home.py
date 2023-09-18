from ib_aitool import app
from flask import Blueprint, render_template
from flask_login import login_required

public_blueprint = Blueprint('front',__name__)

@public_blueprint.route('/front-home')
def index():
    return render_template('home.html', application_list=application_list)

app.register_blueprint(public_blueprint,url_prefix='/')