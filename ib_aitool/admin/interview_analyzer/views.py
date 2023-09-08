from ib_aitool import app
from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import login_required
from ib_aitool.admin.decorators import has_permission
products_blueprint = Blueprint('interview_analyzer', __name__)


@products_blueprint.route('/')
@login_required
@has_permission('Interview Analyzer')
def index():
    return render_template('admin/interview_analyzer/index.html')


app.register_blueprint(products_blueprint, url_prefix='/admin/interview-analyzer')
