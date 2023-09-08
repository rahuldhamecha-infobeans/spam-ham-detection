from ib_aitool.database import db
from ib_aitool.database.models import Role,RolesPermission

class Permission(db.Model):
    __tablename__ = 'permissions'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(64), unique=True, index=True)

    @property
    def __str__(self):
        return self.name

    def __init__(self, name):
        self.name = name

    def roles(self):
        role_permissions = RolesPermission.query.filter_by(permission_id=self.id)
        roles = []
        if role_permissions:
            for role in role_permissions:
                role_data = Role.query.get(role.role_id)
                roles.append(role_data.name)

        return roles