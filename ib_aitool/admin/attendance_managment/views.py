from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify,Response
from flask_login import login_required, current_user
from ib_aitool.database.models.EmployeeModel import Employee, EmployeeAttendance, EmployeeImage
from ib_aitool.admin.attendance_managment.forms import EmployeeForm
from ib_aitool.admin.decorators import has_permission
from ib_aitool import app
from ib_aitool.database import db
from datetime import datetime
import os

attendance_blueprint = Blueprint('attendance', __name__)


@attendance_blueprint.route('/')
@login_required
@has_permission('Attendance')
def index():
    return render_template('admin/attendance/index.html')


@attendance_blueprint.route('/list')
@login_required
@has_permission('Attendance Admin')
def employee_list():
    employee_list = Employee.query.all()
    return render_template('admin/attendance/employee_list.html', employees=employee_list)


@attendance_blueprint.route('/employee-create', methods=['GET', 'POST'])
@login_required
@has_permission('Attendance Admin')
def employee_create():
    form = EmployeeForm()
    if form.validate_on_submit():
        employee = Employee(name=form.name.data)

        db.session.add(employee)
        db.session.commit()
        flash('Employee Successfully Created.', 'success')
        return redirect(url_for('attendance.employee_update', id=employee.id))

    return render_template('admin/attendance/create.html', form=form)


@attendance_blueprint.route('/employee-update/<int:id>', methods=['GET', 'POST'])
@login_required
@has_permission('Attendance Admin')
def employee_update(id):
	employee = Employee.query.get(id)
	form = EmployeeForm(employee_id=id)
	if form.validate_on_submit():
		employee = Employee.query.get(form.employee_id.data)
		if employee:
			employee.name = form.name.data
			db.session.commit()
			flash('Employee Successfully Updated.', 'success')
			return redirect(url_for('attendance.employee_list'))

	return render_template('admin/attendance/create.html', form=form, employee=employee)

@attendance_blueprint.route('/employee/upload-images',methods=['GET','POST'])
@login_required
@has_permission('Attendance Admin')
def upload_images():
    current_date = datetime.now()
    current_time = int(current_date.strftime('%Y%m%d%H%M%S'))
    employee_id = request.form.get('emp_id')
    employee = Employee.query.get(employee_id)
    if employee:
        employee_name = employee.name.lower().replace(' ', '_')

        if 'file' not in request.files:
            return Response('File Not Found.')

        file = request.files['file']

        if file.filename == '':
            return Response('File Is Empty.')
        if file:
            directory = 'employees'
            employee_dir = employee_name
            dir_path = os.path.join(app.config['UPLOAD_FOLDER'], directory,employee_dir)

            filename, file_extension = os.path.splitext(file.filename)
            new_file_name = employee_name + '_' + str(current_time) + file_extension
            isExist = os.path.exists(path=dir_path)
            if not isExist:
                os.makedirs(dir_path)
            file_path = os.path.join(dir_path, new_file_name)
            file.save(file_path)
            file_url = url_for('get_multi_dir_url', dir=directory,dir_2=employee_dir, name=new_file_name)
            new_image = EmployeeImage(image_url=file_url,employee_id=employee.id)
            db.session.add(new_image)
            db.session.commit()
            return Response(file_url)

    return Response('Test')

@attendance_blueprint.route('/employee/image-list')
@login_required
@has_permission('Attendance Admin')
def images_list():
    emp_id = request.args.get('emp_id')
    employee = Employee.query.get(emp_id)
    employee_images = []
    if employee:
        employee_images = employee.images()

    print(employee_images)
    return render_template('admin/attendance/employee_images.html',images=employee_images)

@attendance_blueprint.route('/employee/image/delete/<int:id>')
@login_required
@has_permission('Attendance Admin')
def delete_image(id):
    image = EmployeeImage.query.get(id)
    if image:
        db.session.delete(image)
        db.session.commit()
        return Response('Image Delete Successfully.')
    return Response('Image Not Found.')

@attendance_blueprint.route('/list/delete/<int:id>')
@login_required
@has_permission('Attendance Admin')
def delete_employee(id):
    employee = Employee.query.get(id)
    if employee:
        images = employee.images()
        if images:
            for image in images:
                db.session.delete(image)

        db.session.delete(employee)
        db.session.commit()
        flash('Employee Deleted Successfully.','success')
        return redirect(url_for('attendance.employee_list'))
    return redirect(url_for('attendance.employee_list'))


app.register_blueprint(attendance_blueprint, url_prefix='/admin/attendance')
