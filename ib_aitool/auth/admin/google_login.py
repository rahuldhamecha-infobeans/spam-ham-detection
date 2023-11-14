############# For WITHOUT HTTPS ##################
import os
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
os.environ['OAUTHLIB_RELAX_TOKEN_SCOPE'] = '1'
###################################################

from flask_dance.contrib.google import make_google_blueprint,google
from app_config import google_config
from ib_aitool import app
from flask import render_template,url_for,redirect,request

config = google_config()

google_blueprint = make_google_blueprint(
    client_id=config['GOOGLE_CLIENT_ID'],
    client_secret=config['GOOGLE_CLIENT_SECRET'],
    offline=config['GOOGLE_ENV_LOCAL'],
    scope=config['GOOGLE_SCOPE'],
    redirect_to='auth.google_login_verify'
)

app.register_blueprint(google_blueprint,url_prefix='/admin/auth/login')

@app.route('/admin/auth/login/google')
def google_login():
    if not google.authorized:
        return render_template(url_for('google.login'))

    response = google.get('/oauth2/v1/userinfo?alt=json')
    assert response.ok, response.text

    return redirect(url_for('dashboard.index'))