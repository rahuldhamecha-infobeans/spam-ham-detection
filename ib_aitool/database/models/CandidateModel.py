from ib_aitool.database import db
from ib_aitool.database.models.User import User
from flask_login import current_user

class Candidate(db.Model):
    __tablename__ = 'candidates'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(64),nullable=True,default=None)
    interview_video = db.Column(db.Text,default=None)
    is_report_generated = db.Column(db.Boolean,default=False)
    report_url = db.Column(db.Text,default=None, nullable=True)
    added_by = db.Column(db.Integer, default=0, nullable=True)

    def __str__(self):
        return str(self.name)

    def __init__(self, name,interview_video=None,is_report_generated=False,report_url=None,added_by=0):
        self.name = name
        self.interview_video = interview_video
        self.is_report_generated = is_report_generated
        self.report_url = report_url
        self.added_by = added_by

    def user_data(self):
        user = User.query.get(self.added_by)
        if user:
            return user
        else:
            return None

