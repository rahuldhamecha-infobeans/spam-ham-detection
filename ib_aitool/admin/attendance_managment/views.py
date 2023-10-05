import time

from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, Response, make_response, abort
from flask_login import login_required, current_user
from ib_aitool.database.models.EmployeeModel import Employee, EmployeeAttendance, EmployeeImage
from ib_aitool.admin.attendance_managment.forms import EmployeeForm
from ib_aitool.admin.decorators import has_permission
from ib_aitool import app
from ib_aitool.database import db
from datetime import datetime,date
import os
from ib_aitool.admin.attendance_managment.attendance_model_train import train, predict
from ib_aitool.admin.attendance_managment.camera import Camera
import threading
import cv2
import pytz

attendance_blueprint = Blueprint('attendance', __name__)
camera_instance = None
thread_instnace = None
is_camera_on = False
opened_camera = None
ajax_load = False


@attendance_blueprint.route('/')
@login_required
@has_permission('Attendance')
def index():
    global camera_instance, is_camera_on, opened_camera
    return render_template('admin/attendance/index.html', is_camera_on=is_camera_on,
                           opened_camera=opened_camera)


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


@attendance_blueprint.route('/employee/upload-images', methods=['GET', 'POST'])
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
            dir_path = os.path.join(app.config['UPLOAD_FOLDER'], directory, employee_dir)

            filename, file_extension = os.path.splitext(file.filename)
            new_file_name = employee_name + '_' + str(current_time) + file_extension
            isExist = os.path.exists(path=dir_path)
            if not isExist:
                os.makedirs(dir_path)
            file_path = os.path.join(dir_path, new_file_name)
            file.save(file_path)
            file_url = url_for('get_multi_dir_url', dir=directory, dir_2=employee_dir, name=new_file_name)
            new_image = EmployeeImage(image_url=file_url, employee_id=employee.id)
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

    return render_template('admin/attendance/employee_images.html', images=employee_images)


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
        flash('Employee Deleted Successfully.', 'success')
        return redirect(url_for('attendance.employee_list'))
    return redirect(url_for('attendance.employee_list'))


@attendance_blueprint.route('/list/train-model', methods=['POST'])
@login_required
@has_permission('Attendance Admin')
def employee_train_model():
    dir_path = os.path.join(app.config['UPLOAD_FOLDER'], 'models')
    isExist = os.path.exists(path=dir_path)
    if not isExist:
        os.makedirs(dir_path)

    file_path = os.path.join(dir_path, 'attendance_model.h5')
    train("uploads/employees", model_save_path=file_path, n_neighbors=2)
    return make_response({'status': 'Model Trained Successfully.'}, 200)


@attendance_blueprint.route('/video-feed-for-object')
def video_feed_for_object():
    global camera_instance
    return Response(init_webcam(camera_instance),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@attendance_blueprint.route('/start-camera/<camera_type>')
def start_camera(camera_type):
    if camera_type not in ['entry_camera', 'exit_camera']:
        abort(404)
    global camera_instance, is_camera_on, opened_camera
    if camera_instance and camera_instance.camera != None:
        release_camera()
    is_camera_on = True
    camera_instance = Camera(type=camera_type)
    opened_camera = camera_type

    thread = threading.Thread(target=thread_start, name='Attendance Thread')
    thread.start()
    thread.join(1)
    return redirect(url_for('attendance.index'))


def release_camera():
    global camera_instance, is_camera_on, opened_camera
    if camera_instance != None:
        is_camera_on = False
        opened_camera = None
        camera_instance.release()


@attendance_blueprint.route('/stop-camera/<camera_type>')
def stop_camera(camera_type):
    if camera_type not in ['entry_camera', 'exit_camera']:
        abort(404)
    release_camera()
    return redirect(url_for('attendance.index'))


def init_webcam(camera):
    while True:
        frame = camera.generated_frames()
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')



def thread_start():
    global camera_instance, is_camera_on,ajax_load
    process_this_frame = 59
    model_path = os.path.join(app.config['UPLOAD_FOLDER'], 'models', 'attendance_model.h5')
    if is_camera_on and camera_instance.camera is not None:
        while True:
            ret, frame = camera_instance.camera.read()
            if ret:
                # Different resizing options can be chosen based on desired program runtime.
                # Image resizing for more stable streaming
                img = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                process_this_frame = process_this_frame + 1
                if process_this_frame % 60 == 0:
                    predictions = predict(img, model_path=model_path)
                    if predictions:
                        def is_attendance_existing(employee_id):
                            with app.app_context():
                                employee_attendance = EmployeeAttendance.query.filter_by(employee_id=employee_id,exit_time=None).first()
                                return employee_attendance
                        def store_attendance(predictions):
                            with app.app_context():
                                for name, (top, right, bottom, left) in predictions:
                                    employee_name = name.replace("_", " ")
                                    employee = Employee.query.filter_by(name=employee_name).first()
                                    if employee and employee.id:
                                        current_time = datetime.now(pytz.timezone('Asia/Kolkata'))
                                        existing_attendance = is_attendance_existing(employee.id)
                                        if not existing_attendance and opened_camera == 'entry_camera':
                                            attendance = EmployeeAttendance(entry_time=current_time,exit_time=None,employee_id=employee.id)
                                            db.session.add(attendance)
                                            db.session.commit()
                                        if existing_attendance and opened_camera == 'exit_camera':
                                            attendance = EmployeeAttendance.query.get(existing_attendance.id)
                                            attendance.exit_time = current_time
                                            db.session.commit()
                        store_attendance(predictions)
                        ajax_load = True
                        time.sleep(5)
                        release_camera()


@attendance_blueprint.route('/fetch-attendance-list', methods=['POST'])
def fetch_attendance_list():
    global ajax_load
    # attendance_list = EmployeeAttendance.query.all()
    attendance_list = db.session.query(EmployeeAttendance).filter(EmployeeAttendance.entry_time >= date.today()).order_by(EmployeeAttendance.entry_time.desc()).all()
    html = render_template('admin/attendance/attendance_list.html',attendance_list=attendance_list)
    return {'ajax_reload': ajax_load,'attendance_list' : html}


app.register_blueprint(attendance_blueprint, url_prefix='/admin/attendance')
