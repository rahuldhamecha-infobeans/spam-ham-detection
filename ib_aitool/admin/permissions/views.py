from ib_aitool import app
from ib_aitool.database import db
from flask import Blueprint,render_template,flash,redirect,url_for,abort
from flask_login import login_required
from ib_aitool.database.models.Permission import Permission
from ib_aitool.database.models.RolesPermission import RolesPermission
from ib_aitool.admin.permissions.forms import PermissionForm
from ib_aitool.admin.decorators import has_permission

permissions_blueprint = Blueprint('permissions',__name__)
@permissions_blueprint.route('/')
@login_required
@has_permission('Permissions')
def index():
	permissions = Permission.query.order_by('id').all()
	return render_template('admin/permissions/index.html',permissions=permissions)

@permissions_blueprint.route('/create',methods=['GET','POST'])
@login_required
@has_permission('Permissions Create')
def create():
	form = PermissionForm()
	if form.validate_on_submit():
		permission = Permission(name=form.name.data)

		db.session.add(permission)
		db.session.commit()
		flash('Permission Successfully Created.','success')
		return redirect(url_for('permissions.index'))

	return render_template('admin/permissions/create.html',form=form)


@permissions_blueprint.route('/update/<int:id>',methods=['GET','POST'])
@login_required
@has_permission('Permissions Update')
def update(id):
	permission = Permission.query.get(id)
	form = PermissionForm(permission_id=id)
	if form.validate_on_submit():
		permission = Permission.query.get(form.permission_id.data)
		if permission:
			permission.name = form.name.data
			db.session.commit()
			flash('Permission Successfully Updated.','success')
			return redirect(url_for('permissions.index'))

	return render_template('admin/permissions/create.html',form=form,permission=permission)


@permissions_blueprint.route('/delete/<int:id>',methods=['GET'])
@login_required
@has_permission('Permissions Delete')
def delete(id):
	permission = Permission.query.get(id)
	all_maped_permission = RolesPermission.query.filter_by(permission_id=id)
	if permission:
		if all_maped_permission:
			for mapped in all_maped_permission:
				db.session.delete(mapped)
				db.session.commit()

		db.session.delete(permission)
		db.session.commit()
		flash('Permission Successfully Deleted.','success')
		return redirect(url_for('permissions.index'))

	flash('Permission No Found.', 'error')
	return redirect(url_for('permissions.index'))

app.register_blueprint(permissions_blueprint,url_prefix='/admin/permissions')