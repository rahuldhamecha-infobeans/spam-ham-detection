from flask import Blueprint, render_template,redirect,request,url_for,flash,abort
from ib_aitool import app
from ib_aitool.auth.admin.forms import LoginForm,RegistrationForm
from ib_aitool.database.models.User import User
from flask_login import login_user,logout_user,login_required
from flask_dance.contrib.google import google
from ib_aitool.database import db
from ib_aitool.database.models.Role import Role

auth_admin_blueprint = Blueprint('auth',__name__)

@auth_admin_blueprint.route('/login',methods=['GET','POST'])
def admin_login():
    form = LoginForm()
    # if current_user.is_authenticated:
    #     logout_user()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()

        if user is not None and user.verify_password(form.password.data):
            login_user(user)
            flash('Logged in Successfully!')
            next = request.args.get('next')

            if next == None or not next[0] == '/':
                next = url_for('dashboard.index')

            return redirect(next)
        else:
            flash('Email Or Password Not Matched.')

    return render_template('auth/admin/login.html',form=form)


@auth_admin_blueprint.route('/agent-login',methods=['GET','POST'])
def agent_login():
    form = LoginForm()
    # if current_user.is_authenticated:
    #     logout_user()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()

        if user is not None and user.verify_password(form.password.data):
            login_user(user)
            flash('Logged in Successfully!')
            next = request.args.get('next')

            if next == None or not next[0] == '/':
                next = url_for('home_page')

            return redirect(next)
        else:
            flash('Email Or Password Not Matched.')

    return render_template('auth/admin/agent_login.html',form=form)

@auth_admin_blueprint.route('/register',methods=['GET','POST'])
@login_required
def admin_register():
    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(email=form.email.data,
                    username=form.username.data,
                    password=form.password.data)

        db.session.add(user)
        db.session.commit()
        flash('Thanks for registration!')
        return redirect(url_for('auth.admin_login'))
    return redirect(url_for('auth.admin_login'))
    return render_template('auth/admin/register.html',form=form)


@auth_admin_blueprint.route('/forgot-password.html')
def admin_forgot_password():
    return render_template('auth/admin/forgot-password.html')

@auth_admin_blueprint.route('/logout')
def admin_logout():
    logout_user()
    return redirect(url_for('auth.admin_login'))

@auth_admin_blueprint.route('/google-login-verify')
def google_login_verify():
    response = google.get('/oauth2/v1/userinfo?alt=json')
    assert response.ok, response.text
    google_data = response.json()
    email = google_data['email']
    first_name = google_data['given_name']
    last_name = google_data['family_name']
    full_name = first_name+' '+last_name
    find_user = User.query.filter_by(email=email).first()
    username = email.split('@')[0]
    role = Role.query.filter_by(name='Interviewer').first()
    role_id = 0
    if role:
        role_id = role.id

    if find_user:
        login_user(find_user)
    else:
        new_user = User(email=email,username=username,password=username,full_name=full_name,role_id=role_id)
        db.session.add(new_user)
        db.session.commit()
        login_user(new_user)
    return redirect(url_for('home_page'))


app.register_blueprint(auth_admin_blueprint,url_prefix="/admin/auth")