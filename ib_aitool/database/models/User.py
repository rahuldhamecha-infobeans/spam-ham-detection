from ib_aitool.database import db
from ib_aitool.auth import login_manager
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin
from ib_aitool.database.models.Role import Role


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(user_id)


class User(db.Model, UserMixin):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(64), unique=True, index=True)
    username = db.Column(db.String(64), unique=True, index=True)
    password_hash = db.Column(db.Text)
    role_id = db.Column(db.Integer, default=0, nullable=True)
    full_name = db.Column(db.String(256), default=None, nullable=True)


    def __str__(self):
        return str(self.username)

    def __init__(self, email, username, password,full_name=None,role_id=None):
        self.email = email
        self.username = username
        self.password_hash = generate_password_hash(password)
        self.role_id = role_id
        self.full_name = full_name

    def verify_password(self, password):
        return check_password_hash(self.password_hash, password)

    def role(self):
        if self.role_id:
            role = Role.query.get(self.role_id)
            return role
        else:
            return None

    def role_name(self):
        role = self.role()
        if role:
            return role.name
        else:
            return None
