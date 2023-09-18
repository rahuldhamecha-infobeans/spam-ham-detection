from flask_wtf import FlaskForm
from wtforms import StringField,PasswordField,SubmitField,SelectField,HiddenField
from wtforms.validators import DataRequired,Email,EqualTo
from wtforms import ValidationError
from ib_aitool.database.models.User import User
from ib_aitool.database.models.Role import Role

class UserForm(FlaskForm):
    user_id = HiddenField('User Id')
    full_name = StringField('Full Name',validators=[DataRequired()])
    email = StringField('Email',validators=[DataRequired(),Email()])
    username = StringField('Username',validators=[DataRequired()])
    password = PasswordField('Password')
    roles = SelectField('Role')
    submit = SubmitField('Submit')

    def validate_password(self,password):
        if not self.user_id.data and not self.password.data:
            raise ValidationError('Password Filed is required.')

    def validate_email(self,email):
        if self.user_id.data:
            user = User.query.get(self.user_id.data)
            if user.email != self.email.data and User.query.filter_by(username=self.email.data).first():
                raise ValidationError('Your Email has been already registered.')
        else:
            if User.query.filter_by(username=self.email.data).first():
                raise ValidationError('Your Email has been already registered.')

    def validate_username(self,username):
        if self.user_id.data:
            user = User.query.get(self.user_id.data)
            if user.username != self.username.data and User.query.filter_by(username=self.username.data).first():
                raise ValidationError('Your username has been already registered.')
        else:
            if User.query.filter_by(username=self.username.data).first():
                raise ValidationError('Your username has been already registered.')
