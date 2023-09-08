from ib_aitool import app
from flask_login import current_user


@app.context_processor
def check_permission():
    def user_has_permission(permission):
        role = current_user.role()
        role_permissions = role.permissions()
        if permission not in role_permissions:
            return False
        else:
            return True

    def user_has_role(role_name):
        role = current_user.role()
        if role_name != role.name:
            return False
        else:
            return True

    return {'user_has_permission': user_has_permission,'user_has_role':user_has_role}
