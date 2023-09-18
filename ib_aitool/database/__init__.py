import os
from ib_aitool import app
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from app_config import database_config

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

config = database_config()

database_string = config['USERNAME']+':'+config['PASSWORD']+'@'+config['HOST']+'/'+config['DATABASE']

app.config['SQLALCHEMY_DATABASE_URI'] = "mysql://"+database_string
#use this if above database_uri not working
#app.config['SQLALCHEMY_DATABASE_URI'] = "mysql+mysqldb://"+database_string+"?unix_socket=/run/mysqld/mysqld.sock"
app.config['SQLALCHEMY_TRACK_MODIFICATION'] = False


def create_db():
    db = SQLAlchemy(app)
    return db


db = create_db()
Migrate(app, db)

import ib_aitool.database.register_models
