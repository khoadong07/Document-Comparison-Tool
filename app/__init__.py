import os
from flask import Flask

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)


def create_app():
    from .routes import main as main_blueprint
    app = Flask(__name__,
                template_folder=os.path.join(parent_dir, 'templates'),
                static_folder=os.path.join(parent_dir, 'static'))

    app.register_blueprint(main_blueprint)

    return app
