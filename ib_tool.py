from flask import Flask, render_template


# Create Ecommerce App To Use in multiple files
def create_ecommerce_app():
    app = Flask(__name__)
    return app


app = create_ecommerce_app()
app.config['SECRET_KEY'] = 'infobeans_app_key'



import ib_aitool.register_application
import ib_aitool.context_processor
