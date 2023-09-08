import os
from ib_aitool import app
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

app.config['SQLALCHEMY_DATABASE_URI'] = "mysql://root:Rahul21896@localhost/flask"
app.config['SQLALCHEMY_TRACK_MODIFICATION'] = False


def create_db():
    db = SQLAlchemy(app)
    return db


db = create_db()
Migrate(app, db)

import ib_aitool.database.register_models
