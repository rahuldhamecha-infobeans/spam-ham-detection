from ib_aitool import app
from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import login_required
from ib_aitool.admin.decorators import has_permission
import os
products_blueprint = Blueprint('interview_analyzer', __name__)


@products_blueprint.route('/')
@login_required
@has_permission('Interview Analyzer')
def index():
    return render_template('admin/interview_analyzer/index.html')



def upload_video():
    if 'file' not in request.files:
        raise Exception('file not found.')
    file = request.files['file']
    if file.filename == '':
        raise Exception('No Selected file.')
    if file:
        directory = 'videos'
        dir_path = os.path.join(app.config['UPLOAD_FOLDER'], directory)
        isExist = os.path.exists(path=dir_path)
        if not isExist:
            os.makedirs(dir_path)
        file_path = os.path.join(dir_path, file.filename)
        file.save(file_path)
        file_url = url_for('get_file_url',dir=directory,name=file.filename)
        return file_url
@products_blueprint.route('upload-video',methods=['POST'])
@login_required
@has_permission('Interview Analyzer')
def interview_video_upload():
    if request.method == 'POST':
        video_url = upload_video()
        return video_url
    return 'Test'

app.register_blueprint(products_blueprint, url_prefix='/admin/interview-analyzer')
