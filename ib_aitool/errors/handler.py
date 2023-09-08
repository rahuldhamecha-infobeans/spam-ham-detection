from ib_aitool import app
from flask import render_template

@app.errorhandler(404)
def handle_404(error):
    error_code = 404
    error_title = 'Page Not Found :('
    error_message = "Oops! 😖 The requested URL was not found on this server."
    return render_template('admin/errors/error.html',**locals()), 404


@app.errorhandler(403)
def handle_403(error):
    error_code = 403
    error_title = 'Forbidden'
    error_message = "You don't have the permission to access the requested resource. It is either read-protected or not readable by the server."
    return render_template('admin/errors/error.html',**locals()), 404
