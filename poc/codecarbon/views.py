import os
from flask import render_template, request, render_template, Blueprint
import pandas as pd
from ib_aitool import app
from ib_tool import UPLOAD_FOLDER
import dateutil

codecarbon_blueprint = Blueprint(
    'codecarbon', __name__, template_folder='templates/')


@app.template_filter('strftime')
def _jinja2_filter_datetime(date, fmt=None):
    date = dateutil.parser.parse(date)
    native = date.replace(tzinfo=None)
    format='%b %d, %Y %H:%M:%S'
    return native.strftime(format)


@codecarbon_blueprint.route('/', methods=['POST', 'GET'])
def index():
    project_name = request.args.get('codecarbon_project')
    filename = 'emissions.csv'
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    data_frame = pd.read_csv(filepath)

    project_names = data_frame['project_name'].unique()

    if project_name:
        data_frame = data_frame[data_frame['project_name'] == project_name]

    project_data = data_frame.to_dict(orient='records')
    total_carbon = data_frame['emissions'].sum()
    last_carbon_used = data_frame['emissions'][data_frame['emissions'].tail(1).index].to_numpy()[0]

    total_energy_consumed = data_frame['energy_consumed'].sum()
    last_energy_consumed = data_frame['energy_consumed'][data_frame['energy_consumed'].tail(1).index].to_numpy()[0]

    total_carbon = "{:.2f}".format(total_carbon)
    last_carbon_used = "{:.2f}".format(last_carbon_used)
    total_energy_consumed = "{:.2f}".format(total_energy_consumed)
    last_energy_consumed = "{:.2f}".format(last_energy_consumed)

    temp_args = {
        'project_names': project_names,
        'total_carbon': total_carbon,
        'last_carbon': last_carbon_used,
        'total_power_consumed': total_energy_consumed,
        'last_power_consumed': last_energy_consumed,
        'data_frames' : project_data,
        'selected_project' : project_name
    }

    return render_template('index.html', **temp_args)
