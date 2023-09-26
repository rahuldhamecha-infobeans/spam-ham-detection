from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, IntegerField, HiddenField
from wtforms.validators import DataRequired
from wtforms import ValidationError
from ib_aitool.database.models.EmployeeModel import EmployeeImage,Employee,EmployeeAttendance


class EmployeeForm(FlaskForm):
    employee_id = HiddenField("Employee Id")
    name = StringField('Employee Name', validators=[DataRequired()])
    submit = SubmitField('Submit')
