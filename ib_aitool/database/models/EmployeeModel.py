from ib_aitool.database import db
from flask_login import current_user

class Employee(db.Model):
    __tablename__ = 'employees'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(64),nullable=True,default=None)

    def __str__(self):
        return str(self.name)

    def __init__(self, name):
        self.name = name

    def images(self):
        images = EmployeeImage.query.filter_by(employee_id=self.id)
        return images

    def attendance(self):
        attendance = EmployeeAttendance.query.filter_by(employee_id=self.id)
        return attendance

class EmployeeImage(db.Model):
    __tablename__ = 'employee_images'

    id = db.Column(db.Integer, primary_key=True)
    image_url = db.Column(db.Text, nullable=True, default=None)
    employee_id = db.Column(db.Integer, nullable=True,default=0)


    def __str__(self):
        return str(self.name)

    def __init__(self, image_url,employee_id):
        self.image_url = image_url
        self.employee_id = employee_id

    def employee(self):
        return Employee.query.get(self.employee_id)

class EmployeeAttendance(db.Model):
    __tablename__ = 'employee_attendances'

    id = db.Column(db.Integer, primary_key=True)
    employee_id = db.Column(db.Integer, nullable=True,default=0)
    entry_time = db.Column(db.DateTime, nullable=True, default=None)
    exit_time = db.Column(db.DateTime, nullable=True, default=None)

    def __str__(self):
        return str(self.name)

    def __init__(self, entry_time,exit_time,employee_id):
        self.entry_time = entry_time
        self.employee_id = employee_id
        self.exit_time = exit_time

    def employee(self):
        return Employee.query.get(self.employee_id)