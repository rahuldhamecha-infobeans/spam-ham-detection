from flask_login import current_user
from functools import wraps
from flask import abort

def has_permission(permission=None):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            role = current_user.role()
            role_permissions = role.permissions()
            if permission not in role_permissions:
                abort(403)
            return f(*args,**kwargs)

        return decorated_function
    return decorator



def has_role(role_name=None):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            role = current_user.role()
            if role_name != role.name:
                abort(403)
            return f(*args,**kwargs)

        return decorated_function
    return decorator