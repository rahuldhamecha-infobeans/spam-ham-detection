from ib_aitool import app
from ib_aitool.database import db
from flask import Blueprint,render_template,redirect,flash,url_for,request
from flask_login import login_required
from ib_aitool.admin.roles.forms import RoleForm
from ib_aitool.database.models.Role import Role
from ib_aitool.database.models.Permission import Permission
from ib_aitool.database.models.RolesPermission import RolesPermission
from ib_aitool.admin.decorators import has_permission,has_role

roles_blueprint = Blueprint('roles',__name__)
@roles_blueprint.route('/')
@login_required
@has_permission('Roles')
def index():
	roles = Role.query.order_by('id').all()
	return render_template('admin/roles/index.html',**locals())


@roles_blueprint.route('/create',methods=['GET','POST'])
@login_required
@has_permission('Roles Create')
def create():
	form = RoleForm()
	if form.validate_on_submit():
		role = Role(name=form.name.data)

		db.session.add(role)
		db.session.commit()
		flash('Roles Successfully Created.','success')
		return redirect(url_for('roles.index'))

	return render_template('admin/roles/create.html',form=form)


@roles_blueprint.route('/update/<int:id>',methods=['GET','POST'])
@login_required
@has_permission('Roles Update')
def update(id):
	role = Role.query.get(id)
	form = RoleForm(role_id=id)
	if form.validate_on_submit():
		role = Role.query.get(form.role_id.data)
		if role:
			role.name = form.name.data
			db.session.commit()
			flash('Roles Successfully Updated.','success')
			return redirect(url_for('roles.index'))

	return render_template('admin/roles/create.html',form=form,role=role)


@roles_blueprint.route('/permissions/<int:id>',methods=['GET','POST'])
@login_required
@has_permission('Can Permission Assigned')
@has_role('SuperAdmin')
def permissions(id):
	role = Role.query.get(id)
	permissions = Permission.query.all()
	if request.method == 'POST':
		check_permissions = request.form.getlist('permissions')
		role_id = request.form.get('role_id')

		if check_permissions:
			for permission in check_permissions:
				existing = RolesPermission.query.filter_by(role_id=role_id,permission_id=permission).first()
				if not existing:
					per = RolesPermission(role_id=role_id, permission_id=permission)
					db.session.add(per)
					db.session.commit()

		delete_role_permissions(check_permissions,permissions,role_id)
		flash('Permissions Updated Successfully!','success')

	return render_template('admin/roles/permissions.html',role=role,permissions=permissions)

def delete_role_permissions(check_permissions,permissions,role_id):

	for permission in permissions:
		if str(permission.id) not in check_permissions:
			role_permission = RolesPermission.query.filter_by(role_id=role_id,permission_id=permission.id).first()
			if role_permission:
				db.session.delete(role_permission)
				db.session.commit()

@roles_blueprint.route('/delete/<int:id>',methods=['GET'])
@login_required
@has_permission('Roles Delete')
def delete(id):
	role = Role.query.get(id)
	if role:
		db.session.delete(role)
		db.session.commit()
		flash('Roles Successfully Delete.','success')
		return redirect(url_for('roles.index'))


	flash('Role No Found.', 'error')
	return redirect(url_for('roles.index'))



app.register_blueprint(roles_blueprint,url_prefix='/admin/roles')