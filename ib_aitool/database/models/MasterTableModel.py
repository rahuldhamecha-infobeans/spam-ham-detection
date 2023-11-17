from ib_aitool.database import db
from ib_aitool.database.models.User import User
from flask_login import current_user
from ib_aitool.database.models.CandidateModel import Candidate
from ib_aitool.database.models.TranscriptModel import Transcript

class MasterTable(db.Model):
    __tablename__ = 'mastertable'

    id = db.Column(db.Integer, primary_key=True)
    candidate_table_id = db.Column(db.Integer, db.ForeignKey('candidates.id', ondelete='CASCADE'), nullable=True)
    transcript_table_id = db.Column(db.Integer, db.ForeignKey('transcripts.id', ondelete='CASCADE'), nullable=True)
    type=db.Column(db.String(64),nullable=True,default=None)
    added_by = db.Column(db.Integer, default=0, nullable=True)

    candidate = db.relationship('Candidate', foreign_keys=[candidate_table_id], backref='master_tables',cascade='all, delete')
    transcript = db.relationship('Transcript', foreign_keys=[transcript_table_id], backref='master_tables',cascade='all, delete')
    
    def __str__(self):
        return str(self.name)

    def __init__(self, candidate_table_id=None,transcript_table_id=None,type=None,added_by=0):
        self.candidate_table_id = candidate_table_id
        self.transcript_table_id = transcript_table_id
        self.type = type
        self.added_by = added_by



    def master_table_data(id):
        master_table_data = MasterTable.query.get(id)
        if master_table_data:
            return master_table_data
        else:
            return None