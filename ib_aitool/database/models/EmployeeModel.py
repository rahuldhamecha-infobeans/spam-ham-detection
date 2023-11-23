from ib_aitool.database import db
from flask_login import current_user
from datetime import datetime
import sqlalchemy as sa


class Employee(db.Model):
    __tablename__ = 'employees'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(64), nullable=True, default=None)
    directory_name = db.Column(db.String(64), nullable=True, default=None)

    def __str__(self):
        return str(self.name)

    def __init__(self, name, directory_name=None):
        self.name = name
        self.directory_name = directory_name

    def images(self):
        images = EmployeeImage.query.filter_by(employee_id=self.id)
        return images

    def attendance(self):
        attendance = EmployeeAttendance.query.filter_by(employee_id=self.id)
        return attendance

    def get_attendnace_by_date_group(self):
        attendances = self.attendance()
        date_list = dict()
        if attendances:
            for att in attendances:
                date_time = datetime.strftime(att.entry_time, '%Y-%m-%d')
                if date_time in date_list.keys():
                    date_list[date_time].append(att)
                else:
                    date_list[date_time] = [att]
        return date_list

    def event_list(self):
        date_attendance = self.get_attendnace_by_date_group()
        event_list = []
        if date_attendance:
            for key in date_attendance:
                hours = self.calculate_hours(attendances=date_attendance[key])
                if float(hours) > 6:
                    title = 'Present'
                    background_color = 'green'
                    text_color = 'white'
                else:
                    title = 'Absent'
                    background_color = 'red'
                    text_color = 'white'
                event_data = {
                    'title': title,
                    'start': key,
                    'backgroundColor': background_color,
                    'textColor': text_color
                }
                event_list.append(event_data)
        return event_list

    def calculate_hours(self, attendances):
        hours = 0
        if attendances:
            for att in attendances:
                if att.entry_time != None and att.exit_time != None:
                    t1 = att.entry_time
                    t2 = att.exit_time
                    diff = t2 - t1
                    minutes = diff.total_seconds() / 60
                    minutes = float('{0:.2f}'.format(minutes))
                    hours = minutes / 60
                    hours += float('{0:.2f}'.format(hours))
        return hours


class EmployeeImage(db.Model):
    __tablename__ = 'employee_images'

    id = db.Column(db.Integer, primary_key=True)
    image_url = db.Column(db.Text, nullable=True, default=None)
    employee_id = db.Column(db.Integer, nullable=True, default=0)
    is_image_trained = db.Column(db.Text,nullable=True,default='No')

    def __str__(self):
        return str(self.name)

    def __init__(self, image_url, employee_id):
        self.image_url = image_url
        self.employee_id = employee_id

    def employee(self):
        return Employee.query.get(self.employee_id)


class EmployeeAttendance(db.Model):
    __tablename__ = 'employee_attendances'

    id = db.Column(db.Integer, primary_key=True)
    employee_id = db.Column(db.Integer, nullable=True, default=0)
    entry_time = db.Column(db.DateTime, nullable=True, default=None)
    exit_time = db.Column(db.DateTime, nullable=True, default=None)

    def __str__(self):
        return str(self.name)

    def __init__(self, entry_time, exit_time, employee_id):
        self.entry_time = entry_time
        self.employee_id = employee_id
        self.exit_time = exit_time

    def employee(self):
        return Employee.query.get(self.employee_id)

    def calculate_hours(self):
        hours = 0
        hours_text = ''
        if self.entry_time != None and self.exit_time != None:
            t1 = self.entry_time
            t2 = self.exit_time
            diff = t2 - t1
            minutes = diff.total_seconds() / 60
            minutes = float('{0:.2f}'.format(minutes))
            hours = minutes / 60
            hours += float('{0:.2f}'.format(hours))
            hours_text = str('{0:.2f}'.format(hours)) + ' Hours'
        return {'hours': hours, 'hours_text': hours_text}
