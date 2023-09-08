from ib_aitool import app
from flask import Blueprint,render_template,flash,redirect,url_for
from flask_login import login_required,current_user
from ib_aitool.database.models.User import User
from ib_aitool.database.models.Role import Role
from ib_aitool.admin.users.froms import UserForm
from ib_aitool.database import db
from werkzeug.security import generate_password_hash, check_password_hash
from ib_aitool.admin.decorators import has_permission

user_blueprint = Blueprint('users',__name__)
@user_blueprint.route('/')
@login_required
@has_permission('Users')
def index():
	users = User.query.all()
	return render_template('admin/users/index.html',users=users)


def get_role_choices():
	roles = Role.query.all()
	choices = [(None,'Select Role')]
	choices.extend([(role.id,role.name) for role in roles])
	return choices

@user_blueprint.route('/create',methods=['GET','POST'])
@login_required
@has_permission('Users Create')
def create():
	form = UserForm()
	form.roles.choices = get_role_choices()
	if form.validate_on_submit():
		user = User(
			full_name=form.full_name.data,
			username=form.username.data,
			password=form.password.data,
			email=form.email.data,
			role_id=form.roles.data
		)

		db.session.add(user)
		db.session.commit()
		flash('User Created Successfully!.','success')
		return redirect(url_for('users.index'))

	return render_template('admin/users/create.html',form=form)


@user_blueprint.route('/update/<int:id>',methods=['GET','POST'])
@login_required
@has_permission('Users Update')
def update(id):
	user = User.query.get(id)

	if user:
		form = UserForm(
			user_id=id,
			full_name=user.full_name,
			roles=user.role_id,
			username=user.username,
			email=user.email
		)
	else:
		form = UserForm(user_id=id)

	form.roles.choices = get_role_choices()

	if form.validate_on_submit():
		user_update= User.query.get(form.user_id.data)
		if user_update:
			user_update.full_name = form.full_name.data
			user_update.email = form.email.data
			user_update.role_id = form.roles.data
			user_update.username = form.username.data
			if form.password.data:
				user_update.password_hash = generate_password_hash(form.password.data)

			db.session.commit()
			flash('User Updated Successfully!.','success')
			return redirect(url_for('users.index'))

	return render_template('admin/users/create.html',form=form,user=user)

@user_blueprint.route('/update/profile',methods=['GET','POST'])
@login_required
def profile():
	user = User.query.get(current_user.id)
	is_profile = True
	if user:
		form = UserForm(
			user_id=current_user.id,
			full_name=user.full_name,
			roles=user.role_id,
			username=user.username,
			email=user.email
		)
	else:
		form = UserForm(user_id=id)

	form.roles.choices = get_role_choices()

	if form.validate_on_submit():
		user_update = User.query.get(form.user_id.data)
		if user_update:
			user_update.full_name = form.full_name.data
			user_update.email = form.email.data
			user_update.role_id = form.roles.data
			user_update.username = form.username.data
			if form.password.data:
				user_update.password_hash = generate_password_hash(form.password.data)

			db.session.commit()
			flash('User Updated Successfully!.','success')
			return redirect(url_for('users.profile'))

	return render_template('admin/users/profile.html',form=form,user=user,is_profile=is_profile)

@user_blueprint.route('/delete/<int:id>',methods=['GET'])
@login_required
@has_permission('Users Delete')
def delete(id):
	user = User.query.get(id)
	if user:
		db.session.delete(user)
		db.session.commit()
		flash('User Deleted Successfully.','success')
		return redirect(url_for('users.index'))

	flash('User Not Found.', 'error')
	return redirect(url_for('users.index'))


app.register_blueprint(user_blueprint,url_prefix='/admin/users')