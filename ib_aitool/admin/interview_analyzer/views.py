from ib_aitool import app
from ib_tool import BASE_DIR,mail
from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import login_required,current_user
from ib_aitool.admin.decorators import has_permission
from ib_aitool.database.models.CandidateModel import Candidate
from ib_aitool.database import db
from datetime import datetime
import matplotlib.pyplot as plt
from flask_mail import Message
import pdfkit
import os
import jinja2
products_blueprint = Blueprint('interview_analyzer', __name__)
import subprocess


@products_blueprint.route('/')
@login_required
@has_permission('Interview Analyzer')
def index():
    return render_template('admin/interview_analyzer/index.html')


@products_blueprint.route('/fetch-candidate-list')
def fetch_candidate_list():
    candidates = Candidate.query.filter_by(added_by=current_user.id).order_by('id')
    return render_template('admin/interview_analyzer/candidate_list.html',candidates=candidates)

def upload_video():
    current_date = datetime.now()
    current_time = int(current_date.strftime('%Y%m%d%H%M%S'))
    candidate_name = request.form.get('candidate_name')
    candidate_name = candidate_name.lower().replace(' ','_')

    if 'file' not in request.files:
        return None

    file = request.files['file']
    if file.filename == '':
        return None
    if file:
        directory = 'videos'
        dir_path = os.path.join(app.config['UPLOAD_FOLDER'], directory)

        filename, file_extension = os.path.splitext(file.filename)
        new_file_name = candidate_name+'_'+str(current_time)+file_extension
        isExist = os.path.exists(path=dir_path)
        if not isExist:
            os.makedirs(dir_path)
        file_path = os.path.join(dir_path, new_file_name)
        file.save(file_path)
        file_url = url_for('get_file_url',dir=directory,name=new_file_name)
        return file_url
    return None

@products_blueprint.route('upload-video',methods=['POST'])
@login_required
@has_permission('Interview Analyzer')
def interview_video_upload():
    if request.method == 'POST':
        name = request.form.get('candidate_name')
        video_url = upload_video()
        message = ''
        if name and video_url:
            candidate = Candidate(name=name,interview_video=video_url,added_by=current_user.id)
            db.session.add(candidate)
            db.session.commit()
            message = 'Candidate Added Successfully.'
        else:
            message = 'Please Provide Video and Name.'

        return message
    raise Exception('Invalid Method')


@products_blueprint.route('/generate-report-command')
@login_required
@has_permission('Interview Analyzer')
def generate_report_command():
    candidate_id = request.args.get('candidate')
    command = 'flask generate-report --candidate='+candidate_id
    subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    return redirect(url_for('interview_analyzer.index'))


def get_dir_path(dir):
    dir_path = os.path.join(app.config['UPLOAD_FOLDER'], dir)

    isExist = os.path.exists(path=dir_path)
    if not isExist:
        os.makedirs(dir_path)

    return dir_path

def save_plot_image(candidate_name,data,keys,file_name):
    candidate_path = get_dir_path('reports/')
    overall_filename = candidate_name + file_name
    overall_path = candidate_path + '/' + overall_filename
    overall_url = '/uploads/reports/'+overall_filename
    plt.pie(data, labels=keys, autopct='%.0f%%')
    plt.savefig(overall_path)
    plt.clf()
    return overall_url


def generate_report_pdf(candidate_id):
    candidate = Candidate.query.get(candidate_id)

    templateLoader = jinja2.FileSystemLoader(searchpath="./")
    templateEnv = jinja2.Environment(loader=templateLoader)
    TEMPLATE_FILE = "templates/admin/interview_analyzer/report.html"
    template = templateEnv.get_template(TEMPLATE_FILE)

    data = {
        'overall': {'data': [2, 5, 10, 3, 5], 'keys': ['Sad', 'Nervous', 'Happy', 'Confuse', 'Rocking']},
        'questions' : [
            {'question_no':1,'question' : 'Describe Flask Lifecycle ?','data': [1, 6, 5, 7, 3], 'keys': ['Sad', 'Nervous', 'Happy', 'Confuse', 'Rocking']},
            {'question_no':2,'question' : 'Describe Flask Lifecycle ?','data': [2, 7, 6, 8, 4], 'keys': ['Sad', 'Nervous', 'Happy', 'Confuse', 'Rocking']},
            {'question_no':3,'question' : 'Describe Flask Lifecycle ?','data': [3, 8, 7, 9, 5], 'keys': ['Sad', 'Nervous', 'Happy', 'Confuse', 'Rocking']},
            {'question_no':4,'question' : 'Describe Flask Lifecycle ?','data': [4, 9, 8, 10, 6], 'keys': ['Sad', 'Nervous', 'Happy', 'Confuse', 'Rocking']},
            {'question_no':5,'question' : 'Describe Flask Lifecycle ?','data': [5, 10, 9, 11, 7], 'keys': ['Sad', 'Nervous', 'Happy', 'Confuse', 'Rocking']},
            {'question_no':6,'question' : 'Describe Flask Lifecycle ?','data': [6, 11, 10, 12, 8], 'keys': ['Sad', 'Nervous', 'Happy', 'Confuse', 'Rocking']},
        ]
    }

    template_data = {}
    candidate_name = candidate.name.replace(' ','_').lower()
    # Overall Chart Data
    overall_data = data['overall']['data']
    overall_keys = data['overall']['keys']
    overall_url = save_plot_image(candidate_name,overall_data,overall_keys,'_overall.jpg')

    template_data['overall'] = {'url' : overall_url}

    questions = data['questions']

    current_date = datetime.now()
    current_time = int(current_date.strftime('%Y%m%d%H%M%S'))

    if questions:
        template_data['questions'] = []
        for question in questions:
            file_name = '_question_'+str(current_time)+'_'+str(question['question_no'])+'.jpg'
            question_url = save_plot_image(candidate_name,question['data'],question['keys'],file_name)
            question_data = {
                'question_title' : question['question'],
                'question_url' : question_url
            }
            template_data['questions'].insert(question['question_no'],question_data)

    outputText = template.render(candidate=candidate,report_data = template_data,base_dir=BASE_DIR)

    dir_path = get_dir_path('reports')
    file_name = candidate_name+str(current_time)+'_reports.pdf'
    report_path = dir_path+'/'+file_name
    report_url = '/uploads/reports/'+file_name
    pdfkit.from_string(outputText, report_path,options={"enable-local-file-access": ""})
    remove_files(template_data)
    return report_url


def remove_files(template_data):
    os.remove(BASE_DIR+template_data['overall']['url'])
    questions = template_data['questions']
    if questions:
        for question in questions:
            os.remove(BASE_DIR+question['question_url'])

app.register_blueprint(products_blueprint, url_prefix='/admin/interview-analyzer')
