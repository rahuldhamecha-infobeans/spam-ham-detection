from ib_aitool import app
from flask_login import LoginManager

login_manager = LoginManager()

login_manager.init_app(app)
login_manager.login_view = 'auth.admin_login'

import ib_aitool.auth.admin.views
import ib_aitool.auth.admin.google_login