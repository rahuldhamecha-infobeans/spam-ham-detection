from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, IntegerField, HiddenField
from wtforms.validators import DataRequired
from wtforms import ValidationError
from ib_aitool.database.models.EmployeeModel import EmployeeImage,Employee,EmployeeAttendance
from ib_aitool.database import db


class EmployeeForm(FlaskForm):
    employee_id = HiddenField("Employee Id")
    name = StringField('Employee Name', validators=[DataRequired()])
    submit = SubmitField('Submit')

    # def validate_name(self):
    #     if Employee.query.filter_by(name=self.name.data).first():
    #         raise ValidationError('Employee Already Exists.')