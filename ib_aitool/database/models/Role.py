from ib_aitool.database import db
from ib_aitool.database.models.RolesPermission import RolesPermission
from ib_aitool.database.models.Permission import Permission


class Role(db.Model):
    __tablename__ = 'roles'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(64), unique=True, index=True)

    def __str__(self):
        return str(self.name)

    def __init__(self, name):
        self.name = name

    def permissions(self):
        role_permissions = RolesPermission.query.filter_by(role_id=self.id)
        permissions = []
        if role_permissions:
            for role in role_permissions:
                permission_data = Permission.query.get(role.permission_id)
                permissions.append(permission_data.name)

        return permissions