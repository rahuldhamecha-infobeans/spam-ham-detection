from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, IntegerField, HiddenField
from wtforms.validators import DataRequired
from wtforms import ValidationError
from ib_aitool.database.models.Role import Role


class RoleForm(FlaskForm):
    role_id = HiddenField("Role Id")
    name = StringField('Role Name', validators=[DataRequired()])
    submit = SubmitField('Submit')

    def validate_name(self, name):
        if self.role_id.data:
            role = Role.query.get(self.role_id.data)
            if role.name != self.name.data and Role.query.filter_by(name=self.name.data).first():
                raise ValidationError('Role has been already added.')
        else:
            if Role.query.filter_by(name=self.name.data).first():
                raise ValidationError('Role has been already added.')
