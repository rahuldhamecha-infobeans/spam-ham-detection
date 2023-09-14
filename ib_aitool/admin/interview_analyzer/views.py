import json
import sys
from ib_aitool.database.models.VideoProcessModel import VideoProcess
from ib_aitool.database.models.VideoProcessModel import VideoReport
import subprocess
from ib_aitool import app
from ib_tool import BASE_DIR, mail
from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
from flask_login import login_required, current_user
from ib_aitool.admin.decorators import has_permission
from ib_aitool.database.models.CandidateModel import Candidate
from ib_aitool.database import db
from datetime import datetime
import matplotlib.pyplot as plt
from flask_mail import Message
import pdfkit
import os
import jinja2
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import plotly.io as pio

products_blueprint = Blueprint('interview_analyzer', __name__)


@products_blueprint.route('/')
@login_required
@has_permission('Interview Analyzer')
def index():
    return render_template('admin/interview_analyzer/index.html')


@products_blueprint.route('/fetch-candidate-list')
def fetch_candidate_list():
    candidates = Candidate.query.filter_by(
        added_by=current_user.id).order_by('id')
    return render_template('admin/interview_analyzer/candidate_list.html', candidates=candidates)


def upload_video():
    current_date = datetime.now()
    current_time = int(current_date.strftime('%Y%m%d%H%M%S'))
    candidate_name = request.form.get('candidate_name')
    candidate_name = candidate_name.lower().replace(' ', '_')

    if 'file' not in request.files:
        return None

    file = request.files['file']
    if file.filename == '':
        return None
    if file:
        directory = 'videos'
        dir_path = os.path.join(app.config['UPLOAD_FOLDER'], directory)

        filename, file_extension = os.path.splitext(file.filename)
        new_file_name = candidate_name + '_' + str(current_time) + file_extension
        isExist = os.path.exists(path=dir_path)
        if not isExist:
            os.makedirs(dir_path)
        file_path = os.path.join(dir_path, new_file_name)
        file.save(file_path)
        file_url = url_for('get_file_url', dir=directory, name=new_file_name)
        return file_url
    return None


@products_blueprint.route('upload-video', methods=['POST'])
@login_required
@has_permission('Interview Analyzer')
def interview_video_upload():
    if request.method == 'POST':
        name = request.form.get('candidate_name')
        video_url = upload_video()
        message = ''
        if name and video_url:
            candidate = Candidate(
                name=name, interview_video=video_url, added_by=current_user.id)
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
    generate_report_pdf(candidate_id)

    candidate_data = Candidate.query.filter_by(id=candidate_id).first()
    if candidate_data:
        candidate_data.is_report_generated = True
        db.session.commit()
    return redirect(url_for('interview_analyzer.index'))


def get_dir_path(dir):
    dir_path = os.path.join(app.config['UPLOAD_FOLDER'], dir)

    isExist = os.path.exists(path=dir_path)
    if not isExist:
        os.makedirs(dir_path)

    return dir_path


def save_plot_image(candidate_name, data, keys, file_name):
    candidate_path = get_dir_path('reports/')
    overall_filename = candidate_name + file_name
    overall_path = candidate_path + '/' + overall_filename
    overall_url = '/uploads/reports/' + overall_filename
    plt.pie(data, labels=keys, autopct='%.0f%%')
    plt.savefig(overall_path)
    plt.clf()
    return overall_url


def generate_report_pdf(candidate_id):
    candidate = Candidate.query.get(candidate_id)
    data = get_video_data(candidate_id)
    print(data)
    for video_report, video_process in data:
        generate_pie_chart(
            video_report.video_process_id, video_report.frame_dur_report, video_report.text_dur_report)

    templateLoader = jinja2.FileSystemLoader(searchpath="./")
    templateEnv = jinja2.Environment(loader=templateLoader)
    TEMPLATE_FILE = "templates/admin/interview_analyzer/report.html"
    template = templateEnv.get_template(TEMPLATE_FILE)

    candidate_name = candidate.name.replace(' ', '_').lower()
    current_date = datetime.now()
    current_time = int(current_date.strftime('%Y%m%d%H%M%S'))
    outputText = template.render(
        candidate=candidate, report_data=data, base_dir=BASE_DIR)

    dir_path = get_dir_path('reports')
    file_name = candidate_name + str(current_time) + '_reports.pdf'
    report_path = dir_path + '/' + file_name
    report_url = '/uploads/reports/' + file_name
    pdfkit.from_string(outputText, report_path, options={
        "enable-local-file-access": ""})
    candidate_data = Candidate.query.filter_by(id=candidate_id).first()
    if candidate_data:
        candidate_data.report_url = report_url
        db.session.commit()
    return report_url


def generate_pie_chart(video_process_id, frame_dur_report, text_dur_report):
    frame_data = json.loads(frame_dur_report)

    labels = []
    values = []

    for emotion, value in frame_data['emotions'].items():
        labels.append(emotion)
        values.append(value)

    generate_pie_chart_helper(video_process_id, labels,
                              values, name='_frame_analysis_chart_')

    text_data = json.loads(text_dur_report)

    labels = []
    values = []

    for emotion, value in text_data['emotions'].items():
        labels.append(emotion)
        values.append(value)

    generate_pie_chart_helper(video_process_id, labels,
                              values, name='_text_analysis_chart_')


def generate_pie_chart_helper(video_process_id, labels, values, name):
    video_process_id = str(video_process_id)
    colors = ["#373742", "#E6E6ED", "#EA1B3D", "#676775", "#EB4C5E"]

    # Create a pie chart using Plotly
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Pie(showlegend=False, labels=labels, values=values,
                         textinfo="label+percent", marker=dict(colors=colors), hole=.3))

    chart_image_path = os.path.join(
        app.root_path, 'uploads/reports/graphs', name + video_process_id + '.png')
    pio.write_image(fig, chart_image_path, format='png')


def remove_files(template_data):
    os.remove(BASE_DIR + template_data['overall']['url'])
    questions = template_data['questions']
    if questions:
        for question in questions:
            os.remove(BASE_DIR + question['question_url'])


@products_blueprint.route('/view-reports/<id>')
@login_required
@has_permission('Interview Analyzer')
def view_report(id):
    candidate = Candidate.query.get(id)
    return render_template('admin/interview_analyzer/view_report.html', candidate=candidate)


@products_blueprint.route('/view-video/<id>')
@login_required
@has_permission('Interview Analyzer')
def view_video(id):
    candidate_data = Candidate.query.filter_by(id=id).first()
    print("*****************************")
    print(candidate_data.interview_video)
    print("*********************")

    return render_template(candidate_data.interview_video)


def get_video_data(video_id):
    try:
        query = db.session.query(VideoReport, VideoProcess) \
            .join(VideoProcess, VideoReport.video_process_id == VideoProcess.id) \
            .filter(VideoProcess.vid == video_id)
        data = query.all()
        # Print the SQL query
        print(query.statement)
        return data
    except Exception as e:
        print(f"Error: {e}")
        return None


app.register_blueprint(
    products_blueprint, url_prefix='/admin/smart-interview-assessment')
