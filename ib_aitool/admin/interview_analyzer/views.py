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
from ib_aitool.database.models.VideoProcessModel import VideoProcess
import math
from moviepy.editor import VideoFileClip
from ib_aitool.database import db
from datetime import datetime
import matplotlib.pyplot as plt
from flask_mail import Message
import pdfkit
import os
import jinja2
from ib_aitool.admin.interview_analyzer.generate_video_transcript import generate_transcipt,save_frames_for_timestamps,save_audioclip_timestamps,analyze_timestamp_folder,analyze_audio_timestamps_clips
from ib_aitool.admin.interview_analyzer.save_video_analysis_data import save_videots_report,generate_and_save_overall_video_report

current_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

products_blueprint = Blueprint('interview_analyzer', __name__)
import subprocess
import time
import queue
import threading
import plotly
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import plotly.io as pio


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

def convert_save_audio_file(video_path, dir_path, audio_mp3):
    try:
        # Load the video clip
        video_clip = VideoFileClip(video_path)

        # Extract the audio
        audio_clip = video_clip.audio

        # Create the output directory if it doesn't exist
        output_dir = os.path.dirname(f'{dir_path}/')
        os.makedirs(output_dir, exist_ok=True)
         # Define the output audio file path (MP3 format)
        output_audio_path = f'{dir_path}/{audio_mp3}.mp3'
        # Write the audio to the output file (MP3 format)
        audio_clip.write_audiofile(output_audio_path)
        file_url = url_for('get_file_url', dir='audios', name=f'{audio_mp3}.mp3')
        # Close the video and audio clips
        video_clip.close()
        audio_clip.close()

        # Return True to indicate success
        return file_url, True
    except Exception as e:
        # Handle any exceptions and return False
        print(f"Error: {e}")
        return None, False


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

@products_blueprint.route('/upload_video_file')
@login_required
@has_permission('Interview Analyzer')
def interview_video_upload_file():
    return render_template('admin/interview_analyzer/upload_video_file.html')

@products_blueprint.route('upload-video', methods=['POST'])
@login_required
@has_permission('Interview Analyzer')
def interview_video_upload():
    if request.method == 'POST':
        name = request.form.get('candidate_name')
        video_url = upload_video()
        print(video_url)
        if video_url.startswith('/'):
            video_url = video_url[1:]
        else:
            video_url = video_url
        message = ''
        
        if name and video_url:
            current_date = datetime.now()
            current_time = int(current_date.strftime('%Y%m%d%H%M%S'))
            candidate_name_escaped = name.lower().replace(' ', '_')
            audio_file_name    =      candidate_name_escaped+'_' + str(current_time)
            directory = 'audios'
            dir_path = os.path.join(app.config['UPLOAD_FOLDER'], directory)
            audio_output_path, audio_result = convert_save_audio_file(video_url, dir_path, audio_file_name)
            if audio_result:
                if audio_output_path.startswith('/'):
                    audio_output_path = audio_output_path[1:]
                else:
                    audio_output_path = audio_output_path
                audio_file=audio_output_path
            else:
                audio_file=None

            candidate = Candidate(
                name=name, interview_video=video_url,interview_audio=audio_output_path, added_by=current_user.id)
            db.session.add(candidate)
            db.session.commit()
            message = 'Candidate Added Successfully.'
        else:
            message = 'Please Provide Video and Name.'

        return redirect(url_for('interview_analyzer.index'))
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

def calculate_overall_confidence(facial_emotion_data):

    facial_emotion_data = facial_emotion_data.replace("'", "\"")
    facial_emotion_data = json.loads(facial_emotion_data)

    neutral_percentage = facial_emotion_data['neutral'] * 100
    happy_percentage = facial_emotion_data['happy'] * 100
    fear_percentage = facial_emotion_data['fear'] * 100
    angry_percentage = facial_emotion_data['angry'] * 100
    sad_percentage = facial_emotion_data['sad'] * 100
    surprise_percentage = facial_emotion_data['surprise'] * 100

    if neutral_percentage >= 70 and happy_percentage >= 20:
        weighted_average = 100 - (fear_percentage + angry_percentage + sad_percentage)
    elif (
        happy_percentage <= 20
        and neutral_percentage <= 70
        and angry_percentage > 5
        and fear_percentage > 5
    ):
        weighted_average = 100 - (fear_percentage + angry_percentage + sad_percentage)
    elif angry_percentage >= 40 and fear_percentage >= 30:
        weighted_average = 100 - (fear_percentage + angry_percentage + sad_percentage)
    elif neutral_percentage >= 40 and happy_percentage >= 5:
        weighted_average = 100 - (fear_percentage + angry_percentage + sad_percentage)
    else:
        weighted_average = 55
    weighted_average=weighted_average+surprise_percentage

    CS = (happy_percentage + neutral_percentage+surprise_percentage) - (fear_percentage + sad_percentage)
    NS = fear_percentage + sad_percentage
    CL = CS / (CS + NS)  

    # weighted_average= weighted_average+ facial_emotion_data['surprise']
    # Subtract the percentages of 'angry' and 'fear' emotions
    # Ensure that the overall confidence is within the range of 0% to 100%
    overall_confidence = max(0, min(weighted_average, 100))
    return overall_confidence,CS,NS,CL


def generate_report_pdf(candidate_id):
    candidate = Candidate.query.get(candidate_id)
    data,overall = create_overall_data_by_candidate_id(candidate_id)

    for (video_report, video_process) in data:
        generate_pie_chart(
                video_report.video_process_id, video_report.frame_dur_report, video_report.text_dur_report,video_report.audio_report,overall)

    templateLoader = jinja2.FileSystemLoader(searchpath="./")
    templateEnv = jinja2.Environment(loader=templateLoader)
    TEMPLATE_FILE = "templates/admin/interview_analyzer/report.html"
    template = templateEnv.get_template(TEMPLATE_FILE)

    candidate_name = candidate.name.replace(' ', '_').lower()
    current_date = datetime.now()
    current_time = int(current_date.strftime('%Y%m%d%H%M%S'))
    
    outputText = template.render(
        candidate=candidate,report_data=data, base_dir=BASE_DIR,overall=overall)

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


def create_overall_data_by_candidate_id(candidate_id):
    candidate = Candidate.query.get(candidate_id)
    
    # Create dictionaries to store the values
    interviewer_confidence_dict = {}
    candidate_confidence_dict = {}

    # Calculate and store the values for the interviewer
    overall_interviewer_confidence, CS, NS, CL = calculate_overall_confidence(candidate.overall_interviewer_video_report)
    interviewer_confidence_dict['overall_confidence'] = overall_interviewer_confidence
    interviewer_confidence_dict['CS'] = CS
    interviewer_confidence_dict['NS'] = NS
    interviewer_confidence_dict['CL'] = CL*100

    # Calculate and store the values for the candidate
    overall_candidate_confidence, CS, NS, CL = calculate_overall_confidence(candidate.overall_candidate_video_report)
    candidate_confidence_dict['overall_confidence'] = overall_candidate_confidence
    candidate_confidence_dict['CS'] = CS
    candidate_confidence_dict['NS'] = NS
    candidate_confidence_dict['CL'] = CL*100

    overall = {"candidate_id": str(candidate_id),
        "interviewer_video_report": ib_format_json(data=candidate.overall_interviewer_video_report),
        "candidate_video_report": ib_format_json(data=candidate.overall_candidate_video_report),
        "interviewer_text_report": ib_format_json(data=candidate.overall_interviewer_text_report),
        "candidate_text_report": ib_format_json(data=candidate.overall_candidate_text_report),
        "interviewer_audio_report": ib_format_json(data=candidate.overall_interviewer_audio_report),
        "candidate_audio_report": ib_format_json(data=candidate.overall_candidate_audio_report),
        "overall_interviewer_confidence":interviewer_confidence_dict,
        "overall_candidate_confidence":candidate_confidence_dict,
    }
    
    data = get_video_data(candidate_id)
    return data,overall

def generate_pie_chart(video_process_id, frame_dur_report, text_dur_report,audio_report,overall):
    #Video analysis
    frame_data = ib_format_json(frame_dur_report)
    labels, values = generate_label_value_chart(frame_data)
    generate_pie_chart_helper(labels,
                              values, id=video_process_id, name='_frame_analysis_chart_')
    
    #Text sentiments
    text_data = ib_format_json(text_dur_report)
    desired_order = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

    text_data = {key: text_data[key] for key in desired_order}
    labels, values = generate_label_value_chart(text_data)
    generate_pie_chart_helper(labels,
                              values, id=video_process_id, name='_text_analysis_chart_')
    
    #Audio analysis
    text_data = ib_format_json(audio_report)
    labels, values = generate_label_value_chart(text_data)
    generate_pie_chart_helper(labels,
                              values, id=video_process_id, name='_audio_analysis_chart_')

    
    
    #Overall Interviewer sentiments
    candidate_id = overall['candidate_id']
    
    #video report
    labels, values = generate_label_value_chart(overall['interviewer_video_report'])    
    generate_pie_chart_helper(labels,
                              values, id=candidate_id, name='_overall_interviewer_video_report_')

    #text report
    labels, values = generate_label_value_chart(overall['interviewer_text_report'])    
    generate_pie_chart_helper(labels,
                              values, id=candidate_id, name='_overall_interviewer_text_report_')
    
    #audio report
    labels, values = generate_label_value_chart(overall['interviewer_audio_report'])    
    generate_pie_chart_helper(labels,
                              values, id=candidate_id, name='_overall_interviewer_audio_report_')
    
    #Overall Candidate sentiments

    #video report
    labels, values = generate_label_value_chart(overall['candidate_video_report'])    
    generate_pie_chart_helper(labels,
                              values, id=candidate_id, name='_overall_candidate_video_report_')

    #text report
    labels, values = generate_label_value_chart(overall['candidate_text_report'])    
    generate_pie_chart_helper(labels,
                              values, id=candidate_id, name='_overall_candidate_text_report_')
    
    #audio report
    labels, values = generate_label_value_chart(overall['candidate_audio_report'])    
    generate_pie_chart_helper(labels,
                              values, id=candidate_id, name='_overall_candidate_audio_report_')


def generate_pie_chart_helper( labels, values, id, name):
    _id = str(id)
    colors = ["#373742", "#E6E6ED", "#EA1B3D", "#676775", "#EB4C5E"]

    # Define the directory path
    graph_dir = os.path.join(app.root_path, 'uploads/reports/graphs')

    # Create the directory if it doesn't exist
    if not os.path.exists(graph_dir):
        os.makedirs(graph_dir)

    if len(labels) > 0 and len(values) > 0:
        # Create a pie chart using Plotly
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(go.Pie(showlegend=False, labels=labels, values=values,textinfo="label+percent", marker=dict(colors=colors), hole=.3,textfont_size=17))

        chart_image_path = os.path.join(graph_dir, name + _id + '.svg')
        pio.write_image(fig, chart_image_path, format='svg')

def generate_label_value_chart(data):
    labels = []
    values = []

    if data is not None and data != '':
        for emotion, value in data.items():
            #if value > .01:
            labels.append(emotion.capitalize())
            values.append(value)

        if len(labels) == 0 and len(values) == 0 : 
            for emotion, value in data.items():
                labels.append(emotion.capitalize())
                values.append(value)

    return labels,values
    
def ib_format_json(data):
    if data is None or data == '':
        return data
    data = data.replace("'", "\"")
    data = json.loads(data)
    return data

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
    data,overall = create_overall_data_by_candidate_id(id)

    return render_template('admin/interview_analyzer/view_report.html', candidate=candidate,report_data=data,overall=overall)



def analyze_video(queue,candidate_id):
    with app.app_context():
        data = Candidate.get_video_data(candidate_id)
        candidate_data = Candidate.query.filter_by(id=candidate_id).first()
        if candidate_data:
            candidate_data.video_analysis_status = 'inprogress'
            db.session.commit()
        if data is not None:
            videoPath=data.interview_video
            transcriptJson = generate_transcipt(videoPath)
            # Loop through the data and save it to the database
            for entry in transcriptJson:
                video_entry = VideoProcess(
                    vid=candidate_id,
                    start_duration=math.ceil(float(entry['start'])),
                    end_duration=math.ceil(float(entry['end'])),
                    interview_transcript=entry['transcript'],
                    added_by=data.added_by,
                    created_at=datetime.utcnow(),
                    speaker=entry['speaker'],
                )
                db.session.add(video_entry)
                db.session.commit()
            result= True
        else:
            result= False
        queue.put(result)
        time.sleep(1)  # Simulate some processing time
        print('Part 1 completed')  # Debugging statement


def get_video_frames(queue,candidate_id):
    with app.app_context():
        print("part2  confirmation")
        data = Candidate.get_video_data(candidate_id)
        interviewer_data = VideoProcess.get_transcripts('Interviewer',candidate_id)
        candidate_data = VideoProcess.get_transcripts('candidate',candidate_id)
        if data is not None and interviewer_data is not None and candidate_data is not None:
            videoPath=data.interview_video
            audioPath=data.interview_audio
            # Use os.path.basename to get the file name
            video_name = os.path.basename(videoPath)
            # Remove the file extension if needed
            video_name_without_extension, extension = os.path.splitext(video_name)
            #print("Video Name without Extension:", video_name_without_extension)

            saving_frames_interviewer=save_frames_for_timestamps(f'{videoPath}', interviewer_data, f'uploads/{video_name_without_extension}/interviewer/videoframes/', 'frame')
            saving_frames_candidate=save_frames_for_timestamps(f'{videoPath}', candidate_data, f'uploads/{video_name_without_extension}/candidate/videoframes/', 'frame')
            saving_audioclips_interviewer=save_audioclip_timestamps(f'{audioPath}', interviewer_data, f'uploads/{video_name_without_extension}/interviewer/')
            saving_audioclips_candidate=save_audioclip_timestamps(f'{audioPath}', candidate_data, f'uploads/{video_name_without_extension}/candidate/')

            if saving_frames_interviewer and saving_frames_candidate:
                result= True
            else:
                result= False
        else:
            result= False

        queue.put(result)
        time.sleep(1)  # Simulate some processing time
        print('Part 2 completed')  # Debugging statement


def get_timestamp_emotion(queue,candidate_id):
    with app.app_context():
        data = Candidate.get_video_data(candidate_id)
        if data is not None :
            videoPath=data.interview_video
            # Use os.path.basename to get the file name
            video_name = os.path.basename(videoPath)
            # Remove the file extension if needed
            video_name_without_extension, extension = os.path.splitext(video_name)
            audio_emotions_interviewer={}
            audio_emotions_candidate={}
            overall_timestamp_interviewer=analyze_timestamp_folder(f'uploads/{video_name_without_extension}/interviewer/videoframes/')
            overall_timestamp_candidate=analyze_timestamp_folder(f'uploads/{video_name_without_extension}/candidate/videoframes/')
            save_timestamp_video_report_inteviewer=save_videots_report(overall_timestamp_interviewer,audio_emotions_interviewer)
            save_timestamp_video_report_candidate=save_videots_report(overall_timestamp_candidate,audio_emotions_candidate)

            if save_timestamp_video_report_inteviewer and save_timestamp_video_report_candidate:
                result= True
            else:
                result= False
        else:
            result= False

        queue.put(result)
        time.sleep(1)  # Simulate some processing time
        print('Part 3 completed')  # Debugging statement


def save_overall_report_to_candidate_table(queue,candidate_id):
    with app.app_context():
        if candidate_id:
            overall_interviewer_report=generate_and_save_overall_video_report(candidate_id,'Interviewer')
            overall_candidate_report=generate_and_save_overall_video_report(candidate_id,'candidate')
            if overall_interviewer_report or overall_candidate_report:
                result= True
            else:
                result= False
        else:
            result= False

        queue.put(result)
        time.sleep(1)  # Simulate some processing time
        print('Part 4 completed')  # Debugging statement

@products_blueprint.route('/run_tasks', methods=['GET','POST'])
def run_tasks():
    candidate_id = request.json.get('candidate_id')
    task_queue = queue.Queue()

    # Start the analyze_video thread
    analyze_thread = threading.Thread(target=analyze_video, args=(task_queue, candidate_id))
    analyze_thread.start()

    # Wait for analyze_video to complete and check the result
    analyze_thread.join(timeout=7200)
    confirmation = task_queue.get()
    if confirmation:
        # If confirmation is True, start the get_video_frames thread
        get_frames_thread = threading.Thread(target=get_video_frames, args=(task_queue, candidate_id))
        get_frames_thread.start()

        # Wait for get_video_frames to complete and check the result
        get_frames_thread.join()
        final_result = task_queue.get()
        if final_result:
            get_timestamp_emotion_thread = threading.Thread(target=get_timestamp_emotion, args=(task_queue, candidate_id))
            get_timestamp_emotion_thread.start()

            # Wait for get_video_frames to complete and check the result
            get_timestamp_emotion_thread.join(timeout=7200)
            final_result_2 = task_queue.get()
            if final_result_2:
                get_overall_report_thread = threading.Thread(target=save_overall_report_to_candidate_table, args=(task_queue, candidate_id))
                get_overall_report_thread.start()

                # Wait for get_video_frames to complete and check the result
                get_overall_report_thread.join()
                final_result = task_queue.get()
                if final_result:
                    candidate_data = Candidate.query.filter_by(id=candidate_id).first()
                    if candidate_data:
                        candidate_data.video_analysis_status = 'completed'
                        db.session.commit()
            else:
                final_result=False
    else:
        final_result = False
    
    return jsonify({'result': final_result})

@products_blueprint.route('/view-video/<id>')
@login_required
@has_permission('Interview Analyzer')
def view_video(id):
    candidate_data = Candidate.query.filter_by(id=id).first()
    return render_template(candidate_data.interview_video)


def get_video_data(video_id):
    try:
        query = db.session.query(VideoReport, VideoProcess) \
            .join(VideoProcess, VideoReport.video_process_id == VideoProcess.id) \
            .filter(VideoProcess.vid == video_id)
        query = query.order_by(VideoReport.video_process_id.asc())
        data = query.all()
        return data
    except Exception as e:
        print(f"Error: {e}")
        return None


app.register_blueprint(
    products_blueprint, url_prefix='/admin/smart-interview-assessment')

