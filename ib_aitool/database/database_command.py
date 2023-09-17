from ib_aitool import app
from ib_aitool.database import db
from ib_aitool.database.models.User import User
from ib_aitool.database.models.Role import Role
from ib_aitool.database.models.RolesPermission import RolesPermission
from ib_aitool.database.models.Permission import Permission

# create command

def migrate_user():
    #Create User Data
    role = Role.query.filter_by(name='SuperAdmin').first()
    role_id = 0
    if role:
        role_id = role.id

    user = User(
        email='superadmin@infobeans.com',
        username='superadmin',
        password='SuperAdmin@123',
        role_id=role_id,
        full_name= 'Super Admin'
    )
    db.session.add(user)
    db.session.commit()
    print('Username : superadmin@infobeans.com')
    print('Password : SuperAdmin@123')

def migrate_roles():
    roles = ['SuperAdmin','Interviewer']
    for role in roles:
        role = Role(name=role)
        db.session.add(role)
    db.session.commit()

def migrate_permissions():
    permissions = ['Can Permission Assigned',
                   'Dashboard',
                   'Interview Analyzer',
                   'Permissions',
                   'Permissions Create',
                   'Permissions Update',
                   'Roles',
                   'Roles Create',
                   'Roles Update',
                   'Users',
                   'Users Create',
                   'Users Update'
                 ]
    for perm in permissions:
        permission = Permission(name=perm)
        db.session.add(permission)
    db.session.commit()

def migrate_role_map_permission():
    # For Super Admin
    role = Role.query.filter_by(name='SuperAdmin').first()
    permissions = Permission.query.all()
    if role and permissions:
        for perm in permissions:
            map = RolesPermission(role_id=role.id,permission_id=perm.id)
            db.session.add(map)
        db.session.commit()

    # For Interviewer
    inter_permissions= ['Dashboard','Interview Analyzer']
    role = Role.query.filter_by(name='Interviewer').first()
    if role:
        for perm in inter_permissions:
            permission = Permission.query.filter_by(name=perm).first()
            if permission:
                map = RolesPermission(role_id=role.id,permission_id=permission.id)
                db.session.add(map)
        db.session.commit()


@app.cli.command('migrate-data')
def migrate_data():
    migrate_roles()
    migrate_permissions()
    migrate_role_map_permission()
    migrate_user()

# add command function to cli commands
app.cli.add_command(migrate_data)

