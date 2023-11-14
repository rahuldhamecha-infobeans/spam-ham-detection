from ib_aitool.database import db
from ib_aitool.database.models.User import User
from flask_login import current_user

class Transcript(db.Model):
    __tablename__ = 'transcripts'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(64),nullable=True,default=None)
    transcript = db.Column(db.Text,default=None)
    added_by = db.Column(db.Integer, default=0, nullable=True)
    overall_interviewer_transcript_report=db.Column(db.Text,default=None)
    overall_candidate_transcript_report=db.Column(db.Text,default=None)
    transcript_analysis_status=db.Column(db.String(64),nullable=False,default='pending')
    interviewer=db.Column(db.String(64),nullable=True,default=None)

    def __str__(self):
        return str(self.name)

    def __init__(self, name,transcript=None,added_by=0,overall_interviewer_transcript_report=None,overall_candidate_transcript_report=None,transcript_analysis_status='pending',interviewer=None):
        self.name = name
        self.transcript = transcript
        self.added_by = added_by
        self.overall_interviewer_transcript_report = overall_interviewer_transcript_report
        self.overall_candidate_transcript_report = overall_candidate_transcript_report
        self.transcript_analysis_status = transcript_analysis_status
        self.interviewer = interviewer

    def user_data(self):
        user = User.query.get(self.added_by)
        if user:
            return user
        else:
            return None

    def get_transcript_data(id):
        transcript_data = Transcript.query.get(id)
        if transcript_data:
            return transcript_data
        else:
            return None