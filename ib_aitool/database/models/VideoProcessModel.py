from ib_aitool.database import db
from ib_aitool.database.models.User import User
from flask_login import current_user
from datetime import datetime

# Define the VideoProcess model
class VideoProcess(db.Model):
    __tablename__ = 'video_process'
    id = db.Column(db.Integer, primary_key=True)
    vid = db.Column(db.Integer)
    start_duration = db.Column(db.String(255))
    end_duration = db.Column(db.String(255))
    speaker  = db.Column(db.String(255))
    interview_transcript = db.Column(db.Text)
    added_by = db.Column(db.Integer)
    created_at = db.Column(db.DateTime)

    def __str__(self):
        return str(self.interview_transcript)

    def __init__(self, vid, start_duration, end_duration, interview_transcript, added_by, created_at,speaker):
        self.vid = vid
        self.start_duration = start_duration
        self.end_duration = end_duration
        self.interview_transcript = interview_transcript
        self.added_by = added_by
        self.created_at = created_at
        self.speaker = speaker

    def get_transcripts(speaker,video_id):
        transcript_data = VideoProcess.query.filter_by(speaker=speaker, vid=video_id).all()
        if transcript_data:
            return transcript_data
        else:
            return None
    def get_transcripts_by_videoprocessid(video_process_id):
        transcript_data = VideoProcess.query.get(video_process_id)
        if transcript_data:
            return transcript_data
        else:
            return None
# Define the VideoReport model
class VideoReport(db.Model):
    __tablename__ = 'video_report'
    id = db.Column(db.Integer, primary_key=True)
    video_process_id = db.Column(db.Integer, db.ForeignKey('video_process.id'))
    frame_dur_report = db.Column(db.Text)
    text_dur_report = db.Column(db.Text)
    audio_report = db.Column(db.Text)
    speaker=db.Column(db.String(255))
    video_id = db.Column(db.Integer)
    added_by = db.Column(db.Integer)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __init__(self, video_process_id, frame_dur_report, text_dur_report, audio_report, speaker,video_id, added_by,created_at):
        self.video_process_id = video_process_id
        self.frame_dur_report = frame_dur_report
        self.text_dur_report = text_dur_report
        self.audio_report = audio_report
        self.speaker = speaker
        self.video_id = video_id
        self.added_by = added_by
        self.created_at = created_at
    
    def get_video_timestamp_report(speaker,video_id):
        timestamp_data = VideoReport.query.filter_by(speaker=speaker, video_id=video_id).all()
        if timestamp_data:
            return timestamp_data
        else:
            return None