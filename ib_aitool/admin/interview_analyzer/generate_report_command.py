from ib_aitool import app
from ib_tool import mail
import click
from ib_aitool.database.models.CandidateModel import Candidate
from ib_aitool.database import db
from ib_aitool.admin.interview_analyzer.views import generate_report_pdf
from flask_mail import Message

# create command
@app.cli.command()
@click.option('--candidate')
def generate_report(candidate=None):
    if candidate:
        candidate_data = Candidate.query.get(candidate)
        report_url = generate_report_pdf(candidate)
        if candidate_data:
            candidate_data.is_report_generated = True
            candidate_data.report_url = report_url
            db.session.commit()
    else:
        print('Candidate Not Updated')

# add command function to cli commands
app.cli.add_command(generate_report)