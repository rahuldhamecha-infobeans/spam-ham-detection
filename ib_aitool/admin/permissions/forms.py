from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, IntegerField, HiddenField
from wtforms.validators import DataRequired
from wtforms import ValidationError
from ib_aitool.database.models.Permission import Permission


class PermissionForm(FlaskForm):
    permission_id = HiddenField("Permission Id")
    name = StringField('Permission Name', validators=[DataRequired()])
    submit = SubmitField('Submit')

    def validate_name(self, name):
        if self.permission_id.data:
            permission = Permission.query.get(self.permission_id.data)
            if permission.name != self.name.data and Permission.query.filter_by(name=self.name.data).first():
                raise ValidationError('Permission has been already added.')
        else:
            if Permission.query.filter_by(name=self.name.data).first():
                raise ValidationError('Permission has been already added.')
