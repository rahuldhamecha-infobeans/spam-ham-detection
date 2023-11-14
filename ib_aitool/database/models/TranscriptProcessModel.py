from ib_aitool.database import db
from ib_aitool.database.models.User import User
from flask_login import current_user
from datetime import datetime

# Define the TranscriptProcess model
class TranscriptProcess(db.Model):
    __tablename__ = 'transcript_process'
    id = db.Column(db.Integer, primary_key=True)
    tid = db.Column(db.Integer)
    speaker  = db.Column(db.String(255))
    speaker_type =db.Column(db.String(255))
    interview_transcript = db.Column(db.Text)
    text_dur_report = db.Column(db.Text)
    added_by = db.Column(db.Integer)
    created_at = db.Column(db.DateTime)

    def __str__(self):
        return str(self.interview_transcript)

    def __init__(self, tid, interview_transcript, text_dur_report,added_by, created_at,speaker,speaker_type):
        self.tid = tid
        self.interview_transcript = interview_transcript
        self.text_dur_report = text_dur_report
        self.speaker = speaker
        self.speaker_type = speaker_type
        self.added_by = added_by
        self.created_at = created_at

        

    def get_transcripts(transcript_id):
        transcript_data = TranscriptProcess.query.filter_by(tid=transcript_id).all()
        if transcript_data:
            return transcript_data
        else:
            return None
    def get_transcripts_by_transcriptprocessid(transcript_process_id):
        transcript_data = TranscriptProcess.query.get(transcript_process_id)
        if transcript_data:
            return transcript_data
        else:
            return None
    def get_transcripts_by_speaker_type(transcript_id,speaker_type):
        transcript_data = TranscriptProcess.query.filter_by(tid=transcript_id,speaker_type=speaker_type).all()
        if transcript_data:
            return transcript_data
        else:
            return None
