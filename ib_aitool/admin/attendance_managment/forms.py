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

    def validate_name(self,name):
        if self.employee_id and self.employee_id.data and int(self.employee_id.data) > 0:
            if db.session.query(Employee).filter(Employee.id != self.employee_id.data).filter(Employee.name == self.name.data).first():
                raise ValidationError('Employee Already Exists.')
        elif Employee.query.filter_by(name=self.name.data).first():
            raise ValidationError('Employee Already Exists.')