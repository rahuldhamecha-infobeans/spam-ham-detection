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
import glob
import cv2

from ib_aitool.admin.interview_analyzer.generate_video_transcript import generate_transcipt, save_frames_for_timestamps, \
    save_audioclip_timestamps, analyze_timestamp_folder, analyze_audio_timestamps_clips, transcribe_video, \
    extract_question_timestamps, save_frames_from_video, save_highest_count_videoframe, \
    classify_images_and_generate_timestamp, get_audioclip_timestamps
from ib_aitool.admin.interview_analyzer.save_video_analysis_data import save_videots_report, \
    generate_and_save_overall_video_report
from jinja2 import Environment
import shutil

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
    if str(current_user.role()) == 'SuperAdmin':
        # If the current user is a superadmin, list all candidates
        candidates = Candidate.query.order_by(Candidate.id).all()
    else:
        # If the current user is not a superadmin, list candidates added by them
        candidates = Candidate.query.filter_by(added_by=current_user.id).order_by(Candidate.id).all()

    # candidates = Candidate.query.filter_by(
    #     added_by=current_user.id).order_by('id')
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
            audio_file_name = candidate_name_escaped + '_' + str(current_time)
            directory = 'audios'
            dir_path = os.path.join(app.config['UPLOAD_FOLDER'], directory)
            audio_output_path, audio_result = convert_save_audio_file(video_url, dir_path, audio_file_name)
            if audio_result:
                if audio_output_path.startswith('/'):
                    audio_output_path = audio_output_path[1:]
                else:
                    audio_output_path = audio_output_path
                audio_file = audio_output_path
            else:
                audio_file = None

            candidate = Candidate(
                name=name, interview_video=video_url, interview_audio=audio_output_path, added_by=current_user.id)
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

    # Define weights for selected emotions (neutral, happy, and surprise)
    weight_neutral = 0.7
    weight_happy = 0.2
    weight_surprise = 0.05
    weight_angry = 0.6
    weight_fear = 0.7
    weight_sad = 1
    weight_disgust = 1

    # Calculate the weighted sum of selected emotions
    weighted_sum = (
            facial_emotion_data['neutral'] * weight_neutral +
            facial_emotion_data['happy'] * weight_happy +
            facial_emotion_data['surprise'] * weight_surprise
    )

    # Calculate the sum of remaining emotions (angry, disgust, fear, sad)
    other_emotions_sum = (
            facial_emotion_data['angry'] * weight_angry +
            facial_emotion_data['disgust'] * weight_disgust +
            facial_emotion_data['fear'] * weight_fear +
            facial_emotion_data['sad'] * weight_sad
    )

    # Calculate the nervousness score (sum of remaining emotions)
    nervousness_score = other_emotions_sum

    # Calculate the confidence score (weighted sum of selected emotions)
    confidence_score = weighted_sum

    # Normalize the scores to percentages
    total_score = confidence_score + nervousness_score
    CL = (confidence_score / total_score) * 100
    NS = (nervousness_score / total_score) * 100
    # CS = ((facial_emotion_data['happy']*100) + (facial_emotion_data['neutral']*100)+(facial_emotion_data['surprise']*100))

    return 0, 0, NS, CL


def generate_report_pdf(candidate_id):
    candidate = Candidate.query.get(candidate_id)
    data, overall = create_overall_data_by_candidate_id(candidate_id)

    for (video_report, video_process) in data:
        generate_pie_chart(
            video_report.video_process_id, video_report.frame_dur_report, video_report.text_dur_report,
            video_report.audio_report, overall)

    templateLoader = jinja2.FileSystemLoader(searchpath="./")
    templateEnv = jinja2.Environment(loader=templateLoader)
    TEMPLATE_FILE = "templates/admin/interview_analyzer/report.html"
    template = templateEnv.get_template(TEMPLATE_FILE)

    candidate_name = candidate.name.replace(' ', '_').lower()
    current_date = datetime.now()
    current_time = int(current_date.strftime('%Y%m%d%H%M%S'))

    outputText = template.render(
        candidate=candidate, report_data=data, base_dir=BASE_DIR, overall=overall)

    dir_path = get_dir_path('reports')
    file_name = candidate_name + str(current_time) + '_reports.pdf'
    report_path = dir_path + '/' + file_name
    report_url = 'uploads/reports/' + file_name
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
    interviewer_confidence_dict_text = {}
    candidate_confidence_dict = {}
    candidate_confidence_dict_text = {}

    # Calculate and store the values for the interviewer video
    overall_interviewer_confidence, CS, NS, CL = calculate_overall_confidence(
        candidate.overall_interviewer_video_report)
    interviewer_confidence_dict['overall_confidence'] = overall_interviewer_confidence
    interviewer_confidence_dict['CS'] = CS
    interviewer_confidence_dict['NS'] = NS
    interviewer_confidence_dict['CL'] = CL

    # Calculate and store the values for the interviewer text analysis
    overall_interviewer_confidence_text, CS, NS, CL = calculate_overall_confidence(
        candidate.overall_interviewer_text_report)
    interviewer_confidence_dict_text['overall_confidence'] = overall_interviewer_confidence_text
    interviewer_confidence_dict_text['CS'] = CS
    interviewer_confidence_dict_text['NS'] = NS
    interviewer_confidence_dict_text['CL'] = CL

    # Calculate and store the values for the candidate
    overall_candidate_confidence, CS, NS, CL = calculate_overall_confidence(candidate.overall_candidate_video_report)
    candidate_confidence_dict['overall_confidence'] = overall_candidate_confidence
    candidate_confidence_dict['CS'] = CS
    candidate_confidence_dict['NS'] = NS
    candidate_confidence_dict['CL'] = CL

    # Calculate and store the values for the candidate text
    overall_candidate_confidence_text, CS, NS, CL = calculate_overall_confidence(
        candidate.overall_candidate_text_report)
    candidate_confidence_dict_text['overall_confidence'] = overall_candidate_confidence_text
    candidate_confidence_dict_text['CS'] = CS
    candidate_confidence_dict_text['NS'] = NS
    candidate_confidence_dict_text['CL'] = CL

    overall = {"candidate_id": str(candidate_id),
               "interviewer_video_report": ib_format_json(data=candidate.overall_interviewer_video_report),
               "candidate_video_report": ib_format_json(data=candidate.overall_candidate_video_report),
               "interviewer_text_report": ib_format_json(data=candidate.overall_interviewer_text_report),
               "candidate_text_report": ib_format_json(data=candidate.overall_candidate_text_report),
               "interviewer_audio_report": ib_format_json(data=candidate.overall_interviewer_audio_report),
               "candidate_audio_report": ib_format_json(data=candidate.overall_candidate_audio_report),
               "overall_interviewer_confidence": interviewer_confidence_dict,
               "overall_candidate_confidence": candidate_confidence_dict,
               "overall_interviewer_confidence_text": interviewer_confidence_dict_text,
               "overall_candidate_confidence_text": candidate_confidence_dict_text,
               }

    data = get_video_data(candidate_id)
    return data, overall


def generate_pie_chart(video_process_id, frame_dur_report, text_dur_report, audio_report, overall):
    # Video analysis
    frame_data = ib_format_json(frame_dur_report)
    labels, values = generate_label_value_chart(frame_data)
    generate_pie_chart_helper(labels,
                              values, id=video_process_id, name='_frame_analysis_chart_')

    # Text sentiments
    text_data = ib_format_json(text_dur_report)
    desired_order = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

    text_data = {key: text_data[key] for key in desired_order}
    labels, values = generate_label_value_chart(text_data)
    generate_pie_chart_helper(labels,
                              values, id=video_process_id, name='_text_analysis_chart_')

    # Audio analysis
    text_data = ib_format_json(audio_report)
    labels, values = generate_label_value_chart(text_data)
    generate_pie_chart_helper(labels,
                              values, id=video_process_id, name='_audio_analysis_chart_')

    # Overall Interviewer sentiments
    candidate_id = overall['candidate_id']

    # video report
    labels, values = generate_label_value_chart(overall['interviewer_video_report'])
    generate_pie_chart_helper(labels,
                              values, id=candidate_id, name='_overall_interviewer_video_report_')

    # text report
    labels, values = generate_label_value_chart(overall['interviewer_text_report'])
    generate_pie_chart_helper(labels,
                              values, id=candidate_id, name='_overall_interviewer_text_report_')

    # audio report
    labels, values = generate_label_value_chart(overall['interviewer_audio_report'])
    generate_pie_chart_helper(labels,
                              values, id=candidate_id, name='_overall_interviewer_audio_report_')

    # Overall Candidate sentiments

    # video report
    labels, values = generate_label_value_chart(overall['candidate_video_report'])
    generate_pie_chart_helper(labels,
                              values, id=candidate_id, name='_overall_candidate_video_report_')

    # text report
    labels, values = generate_label_value_chart(overall['candidate_text_report'])
    generate_pie_chart_helper(labels,
                              values, id=candidate_id, name='_overall_candidate_text_report_')

    # audio report
    labels, values = generate_label_value_chart(overall['candidate_audio_report'])
    generate_pie_chart_helper(labels,
                              values, id=candidate_id, name='_overall_candidate_audio_report_')


def generate_pie_chart_helper(labels, values, id, name):
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
        fig.add_trace(
            go.Pie(showlegend=False, labels=labels, values=values, textinfo="label+percent", marker=dict(colors=colors),
                   hole=.3, textfont_size=17))

        chart_image_path = os.path.join(graph_dir, name + _id + '.svg')
        pio.write_image(fig, chart_image_path, format='svg')


def generate_label_value_chart(data):
    labels = []
    values = []

    if data is not None and data != '':
        for emotion, value in data.items():
            # if value > .01:
            labels.append(emotion.capitalize())
            values.append(value)

        if len(labels) == 0 and len(values) == 0:
            for emotion, value in data.items():
                labels.append(emotion.capitalize())
                values.append(value)

    return labels, values


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


def text_analysis(paragraph):
    weak_words = [
        'Absolutely', 'Definitely', 'Totally', 'Actually', 'Personally', 'Technically',
        'Virtually', 'Simply', 'Possibly', 'Somehow', 'Just', 'Very', 'Pretty', 'Some', 'Honestly',
        'That', 'Extremely', 'Really', 'Much', 'Exactly', 'Ultimate', 'Complete', 'World-class',
        'Amazing'
    ]

    filler_words = [
        'Well', 'um', 'uh', 'umm', 'hmm', 'hmmm', 'Like', 'Basically', 'seriously',
        'literally', 'totally', 'Clearly', 'you see', 'you know', 'I mean', 'You know what I mean',
        'At the end of the day', 'Believe me', 'Okay'
    ]

    founded_weak_words = []
    founded_filler_words = []
    paragraph_lower = paragraph.lower()

    for word in weak_words:
        lower_word = word.lower()
        count = paragraph_lower.count(lower_word)
        if int(count) > 0:
            data = {'word': word, 'count': count}
            founded_weak_words.append(data)

    for word in filler_words:
        lower_word = word.lower()
        count = paragraph_lower.count(lower_word)
        if int(count) > 0:
            data = {'word': word, 'count': count}
            founded_filler_words.append(data)

    return {'weak_words': founded_weak_words, 'filler_words': founded_filler_words}


@products_blueprint.route('/view-reports/<id>')
@login_required
@has_permission('Interview Analyzer')
def view_report(id):
    candidate = Candidate.query.get(id)
    data, overall = create_overall_data_by_candidate_id(id)
    analysis = get_text_analysis_data(data)
    return render_template('admin/interview_analyzer/view_report.html', candidate=candidate, report_data=data,
                           overall=overall, analysis_data=analysis)


def get_text_analysis_data(data):
    all_text = []
    candidate_text = []
    final_text = ''
    candidate_text_string = ''
    for video_report, video_process in data:
        if video_process.speaker == 'interviewer':
            all_text.append(video_process.interview_transcript)

        if video_process.speaker == 'candidate':
            candidate_text.append(video_process.interview_transcript)

    if len(all_text) > 0:
        final_text = " ".join(all_text)

    if len(candidate_text) > 0:
        candidate_text_string = " ".join(candidate_text)

    data = [('Candidate Analysis', text_analysis(candidate_text_string)),
            ('Interviewer Analysis', text_analysis(final_text))]
    return data


def is_directory_not_empty(directory_path):
    return len(os.listdir(directory_path)) > 0


def analyze_video(queue, candidate_id, selected_image):
    with app.app_context():
        data = Candidate.get_video_data(candidate_id)
        candidate_data = Candidate.query.filter_by(id=candidate_id).first()
        if candidate_data:
            candidate_data.video_analysis_status = 'inprogress'
            db.session.commit()
        if data is not None:
            videoPath = data.interview_video
            audioPath = data.interview_audio
            video_name = os.path.basename(videoPath)
            # Remove the file extension if needed
            video_name_without_extension, extension = os.path.splitext(video_name)
            # transcriptJson = generate_transcipt(videoPath)
            all_frame_count = save_frames_from_video(videoPath, None,
                                                     f'uploads/{video_name_without_extension}/allframes')
            # image_directory_path=output_folder

            output_directory_path = os.path.join(BASE_DIR,
                                                 'uploads/' + video_name_without_extension + '/video-interviewer/')
            selected_image = os.path.join(BASE_DIR + selected_image)
            os.makedirs(output_directory_path, exist_ok=True)
            # Define the custom name and path for the saved image
            custom_image_name = f'{video_name_without_extension}.jpg'
            custom_image_path = os.path.join(output_directory_path, custom_image_name)

            shutil.copy(selected_image, custom_image_path)
            image_dir = f'uploads/{video_name_without_extension}/allframes'
            interviewer_image_path = f'uploads/{video_name_without_extension}/video-interviewer/{video_name_without_extension}.jpg'
            # Check if both the image directory and interviewer image exist and are not empty
            if os.path.exists(image_dir) and os.path.exists(interviewer_image_path) and is_directory_not_empty(
                    image_dir):
                result = classify_images_and_generate_timestamp(image_dir, interviewer_image_path)
                temp_storage_dir_path = f'uploads/{video_name_without_extension}/'
                final_transcript_data = process_video_and_transcript(result, audioPath, temp_storage_dir_path)
                print(final_transcript_data)
                # Loop through the data and save it to the database
                for entry in final_transcript_data:
                    label = list(entry.keys())[0]
                    if entry[label]['transcript_data']:
                        transcript_data = entry[label]['transcript_data']
                    else:
                        transcript_data = 'NA'
                    video_entry = VideoProcess(
                        vid=candidate_id,
                        start_duration=math.ceil(float(entry[label]['start'])),
                        end_duration=math.ceil(float(entry[label]['end'])),
                        interview_transcript=transcript_data,
                        added_by=data.added_by,
                        created_at=datetime.utcnow(),
                        speaker=label,
                    )
                    db.session.add(video_entry)
                    db.session.commit()
            result = True
        else:
            result = False
        print(f'result: {result}')
        queue.put(result)
        time.sleep(1)  # Simulate some processing time
        print('Part 1 completed')  # Debugging statement


@products_blueprint.route('/confirm_interviewer', methods=['POST'])
def confirm_interviewer():
    data = Candidate.get_video_data(request.json.get('candidate_id'))
    if data is not None:
        videoPath = data.interview_video
        video_name = os.path.basename(videoPath)
        video_name_without_extension, extension = os.path.splitext(video_name)
        output_folder = f'uploads/{video_name_without_extension}/two-minutes-videoframes/'
        output_directory_path = f'uploads/{video_name_without_extension}/final-frames/'

        if (request.json.get('sendCandidate')):
            return json.dumps({'success': 1, 'image': output_directory_path + 'candidate.jpg'})
        else:
            timestamp = []
            timestamp.append({'start': 30, 'end': 150})
            save_two_minutes_frame(videoPath, timestamp, output_folder)  # save two minutes frames
            finalize_interviewer_saved = save_highest_count_videoframe(output_folder, output_directory_path)
            if finalize_interviewer_saved:
                return json.dumps({'success': 1, 'image': output_directory_path + 'interviewer.jpg'})
            else:
                return json.dumps({'success': 0})


def save_two_minutes_frame(videoPath, timestamp, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(videoPath)
    frame_count = 0
    start_time = int(timestamp[0]['start'])
    end_time = int(timestamp[0]['end'])
    cap.set(cv2.CAP_PROP_POS_MSEC, (start_time * 1000))
    frame_interval_ms = 1000  # One frame per second (1000 milliseconds)
    while True:
        ret, frame = cap.read()
        if not ret or cap.get(cv2.CAP_PROP_POS_MSEC) > (end_time * 1000):
            break
        frame_filename = f"{output_folder}/frame_{frame_count}.jpg"
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

        # Skip frames to maintain one frame per second
        cap.set(cv2.CAP_PROP_POS_MSEC, frame_count * frame_interval_ms)

    cap.release()


def process_video_and_transcript(result, audio_path, temp_storage_dir_path):
    for segment in result:
        label = list(segment.keys())[0]
        if label == 'candidate':
            transcript_data = get_audioclip_timestamps(audio_path, segment[label]['start'], segment[label]['end'],
                                                       temp_storage_dir_path)
            segment[label]['transcript_data'] = transcript_data
        elif label == 'interviewer':
            transcript_data = get_audioclip_timestamps(audio_path, segment[label]['start'], segment[label]['end'],
                                                       temp_storage_dir_path)
            segment[label]['transcript_data'] = transcript_data

    return result


def get_video_frames(queue, candidate_id):
    with app.app_context():
        print("part2  confirmation")
        data = Candidate.get_video_data(candidate_id)
        interviewer_data = VideoProcess.get_transcripts('Interviewer', candidate_id)
        candidate_data = VideoProcess.get_transcripts('candidate', candidate_id)
        if data is not None and interviewer_data is not None and candidate_data is not None:
            videoPath = data.interview_video
            audioPath = data.interview_audio
            # Use os.path.basename to get the file name
            video_name = os.path.basename(videoPath)
            # Remove the file extension if needed
            video_name_without_extension, extension = os.path.splitext(video_name)
            # print("Video Name without Extension:", video_name_without_extension)

            saving_frames_interviewer = save_frames_for_timestamps(f'{videoPath}', interviewer_data,
                                                                   f'uploads/{video_name_without_extension}/interviewer/videoframes/',
                                                                   'frame')
            saving_frames_candidate = save_frames_for_timestamps(f'{videoPath}', candidate_data,
                                                                 f'uploads/{video_name_without_extension}/candidate/videoframes/',
                                                                 'frame')
            saving_audioclips_interviewer = save_audioclip_timestamps(f'{audioPath}', interviewer_data,
                                                                      f'uploads/{video_name_without_extension}/interviewer/')
            saving_audioclips_candidate = save_audioclip_timestamps(f'{audioPath}', candidate_data,
                                                                    f'uploads/{video_name_without_extension}/candidate/')

            if saving_frames_interviewer and saving_frames_candidate:
                result = True
            else:
                result = False
        else:
            result = False

        queue.put(result)
        time.sleep(1)  # Simulate some processing time
        print('Part 2 completed')  # Debugging statement


def get_timestamp_emotion(queue, candidate_id):
    with app.app_context():
        data = Candidate.get_video_data(candidate_id)
        if data is not None:
            videoPath = data.interview_video
            # Use os.path.basename to get the file name
            video_name = os.path.basename(videoPath)
            # Remove the file extension if needed
            video_name_without_extension, extension = os.path.splitext(video_name)
            audio_emotions_interviewer = {}
            audio_emotions_candidate = {}
            overall_timestamp_interviewer = analyze_timestamp_folder(
                f'uploads/{video_name_without_extension}/interviewer/videoframes/')
            overall_timestamp_candidate = analyze_timestamp_folder(
                f'uploads/{video_name_without_extension}/candidate/videoframes/')
            save_timestamp_video_report_inteviewer = save_videots_report(overall_timestamp_interviewer,
                                                                         audio_emotions_interviewer)
            save_timestamp_video_report_candidate = save_videots_report(overall_timestamp_candidate,
                                                                        audio_emotions_candidate)
            if save_timestamp_video_report_inteviewer and save_timestamp_video_report_candidate:
                result = True
            else:
                result = False
        else:
            result = False

        queue.put(result)
        time.sleep(1)  # Simulate some processing time
        print('Part 3 completed')  # Debugging statement


def save_overall_report_to_candidate_table(queue, candidate_id):
    with app.app_context():
        if candidate_id:
            overall_interviewer_report = generate_and_save_overall_video_report(candidate_id, 'Interviewer')
            overall_candidate_report = generate_and_save_overall_video_report(candidate_id, 'candidate')
            if overall_interviewer_report or overall_candidate_report:
                result = True
            else:
                result = False
        else:
            result = False

        queue.put(result)
        time.sleep(1)  # Simulate some processing time
        print('Part 4 completed')  # Debugging statement


@products_blueprint.route('/run_tasks_modify', methods=['GET', 'POST'])
def run_tasks_modify():
    candidate_id = request.json.get('candidate_id')
    selected_image = request.json.get('selected_image')
    task_queue = queue.Queue()

    # Start the analyze_video thread
    analyze_thread = threading.Thread(target=analyze_video, args=(task_queue, candidate_id, selected_image))
    analyze_thread.start()
    # Wait for analyze_video to complete and check the result
    analyze_thread.join(timeout=7200)
    confirmation = task_queue.get()
    if confirmation:
        final_result = task_queue.get()
    return jsonify({'result': final_result})


@products_blueprint.route('/run_tasks', methods=['GET', 'POST'])
def run_tasks():
    candidate_id = request.json.get('candidate_id')
    selected_image = request.json.get('selected_image')
    task_queue = queue.Queue()

    # Start the analyze_video thread
    analyze_thread = threading.Thread(target=analyze_video, args=(task_queue, candidate_id, selected_image))
    analyze_thread.start()

    # Wait for analyze_video to complete and check the result
    analyze_thread.join(timeout=7200)
    confirmation = task_queue.get()
    if confirmation:
        print("part-1 confirmation")
        # If confirmation is True, start the get_video_frames thread
        get_frames_thread = threading.Thread(target=get_video_frames, args=(task_queue, candidate_id))
        get_frames_thread.start()

        # Wait for get_video_frames to complete and check the result
        get_frames_thread.join()
        final_result = task_queue.get()
        if final_result:
            get_timestamp_emotion_thread = threading.Thread(target=get_timestamp_emotion,
                                                            args=(task_queue, candidate_id))
            get_timestamp_emotion_thread.start()

            # Wait for get_video_frames to complete and check the result
            get_timestamp_emotion_thread.join(timeout=7200)
            final_result_2 = task_queue.get()
            if final_result_2:
                get_overall_report_thread = threading.Thread(target=save_overall_report_to_candidate_table,
                                                             args=(task_queue, candidate_id))
                get_overall_report_thread.start()

                # Wait for get_video_frames to complete and check the result
                get_overall_report_thread.join()
                final_result = task_queue.get()
                if final_result:
                    candidate_data = Candidate.query.filter_by(id=candidate_id).first()
                    if candidate_data:
                        candidate_data.video_analysis_status = 'completed'
                        db.session.commit()
                    c_data = Candidate.get_video_data(candidate_id)
                    remove_all_model_created_files(c_data.interview_video)
            else:
                final_result = False
    else:
        final_result = False

    return jsonify({'result': final_result})


@products_blueprint.route('/view-video/<id>')
@login_required
@has_permission('Interview Analyzer')
def view_video(id):
    candidate_data = Candidate.query.filter_by(id=id).first()
    return render_template(candidate_data.interview_video)


def remove_all_model_created_files(videopath):
    # current_dir = os.getcwd()
    # Extract the base name (without extension) from the video file path
    video_file_path = videopath
    base_name = os.path.splitext(os.path.basename(video_file_path))[0]
    folder_path = video_file_path

    # List all files in the folder
    all_files = glob.glob(os.path.join(BASE_DIR, '*'))
    video_name = os.path.basename(video_file_path)
    # Remove the file extension if needed
    video_name_without_extension, extension = os.path.splitext(video_name)
    video_name_without_extension = f'uploads/{video_name_without_extension}'
    shutil.rmtree(video_name_without_extension)
    # Iterate through the files and delete those with matching base names
    for file_path in all_files:
        print(file_path)
        file_base_name, file_extension = os.path.splitext(os.path.basename(file_path))
        if file_base_name.startswith(base_name):
            os.remove(file_path)
            print(f"Deleted file: {file_path}")
    return jsonify({'result': True})


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


def calculate_qna_confidence(facial_emotion_data):
    facial_emotion_data = facial_emotion_data.replace("'", "\"")
    facial_emotion_data = json.loads(facial_emotion_data)

    weight_neutral = 0.7
    weight_happy = 0.2
    weight_surprise = 0.05
    weight_angry = 0.6
    weight_fear = 0.7
    weight_sad = 1
    weight_disgust = 1

    # Calculate the weighted sum of selected emotions
    weighted_sum = (
            facial_emotion_data['neutral'] * weight_neutral +
            facial_emotion_data['happy'] * weight_happy +
            facial_emotion_data['surprise'] * weight_surprise
    )

    # Calculate the sum of remaining emotions (angry, disgust, fear, sad)
    other_emotions_sum = (
            facial_emotion_data['angry'] * weight_angry +
            facial_emotion_data['disgust'] * weight_disgust +
            facial_emotion_data['fear'] * weight_fear +
            facial_emotion_data['sad'] * weight_sad
    )

    # Calculate the nervousness score (sum of remaining emotions)
    nervousness_score = other_emotions_sum

    # Calculate the confidence score (weighted sum of selected emotions)
    confidence_score = weighted_sum

    # Normalize the scores to percentages
    total_score = confidence_score + nervousness_score
    CL = 'NA'
    NS = 'NA'
    if total_score != 0.0:
        CL = (confidence_score / total_score) * 100
        NS = (nervousness_score / total_score) * 100
        # CS = ((facial_emotion_data['happy']*100) + (facial_emotion_data['neutral']*100)+(facial_emotion_data['surprise']*100))
        return f"CS:{round(CL, 2)}%, NS:{round(NS, 2)}%"
    return f"CS:{CL}, NS:{NS}"


# Register the custom Jinja2 filter
app.jinja_env.filters['emotion_scores'] = calculate_qna_confidence


@app.route('/delete_route/<int:item_id>')
def delete_route(item_id):
    try:
        candidate = Candidate.query.get(item_id)
        video_url = candidate.interview_video
        video_pdf = candidate.report_url
        video_audio = candidate.interview_audio
        video_name = os.path.basename(video_url)
        # Remove the file extension if needed
        video_name_without_extension, extension = os.path.splitext(video_name)
        video_name_without_extension = f'uploads/{video_name_without_extension}'
        # Delete the item from each table
        with db.session.begin_nested():
            db.session.query(VideoReport).filter(VideoReport.video_id == item_id).delete()
            db.session.query(VideoProcess).filter(VideoProcess.vid == item_id).delete()
            db.session.query(Candidate).filter(Candidate.id == item_id).delete()

        # Commit the transaction
        if os.path.exists(video_url):
            os.remove(video_url)
            os.remove(video_pdf)
            os.remove(video_audio)
            if os.path.exists(video_name_without_extension):
                shutil.rmtree(video_name_without_extension)

        db.session.commit()

    except Exception as e:
        # Handle any exceptions that may occur during deletion
        db.session.rollback()
        return f"Error: {str(e)}", 500

    return redirect(url_for('interview_analyzer.index'))


app.register_blueprint(
    products_blueprint, url_prefix='/smart-interview-assessment')
