import os
from flask import Flask, request, url_for, flash, redirect
from flask_restful import Api
from flask_cors import CORS
from werkzeug.utils import secure_filename
from os import environ

from flask import render_template

from demo.resources.asc_task import AscTaskResource, AscTaskListResource

ALLOWED_EXTENSIONS = {'wav'}

def custom_create_app(config: dict = None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    CORS(app)

    app.logger.info("initial flask app")
    # update the config
    if config is None:
        app.config.from_mapping({
            "UPLOAD_FOLDER": os.path.abspath("demo/static/upload")
        })
    else:
        app.config.from_mapping(**config)

    api = Api(app)
    api.add_resource(AscTaskListResource, "/asc-tasks")
    api.add_resource(AscTaskResource, "/asc-tasks/<asc_task_id>")

    # ma.init_app(app)

    @app.before_request
    def log_request_info():
        app.logger.info('Body: %s', request.get_data())

    # @app.route('/hello/')
    # @app.route('/hello/<name>')
    # def hello(name=None):
    #     return render_template('hello.html', name=name)

    @app.route('/')
    def demo(name=None):
        return render_template('demo.html')

    def allowed_file(filename):
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    @app.route('/upload', methods=['POST'])
    def upload_file():
        if request.method == 'POST':
            # check if the post request has the file part
            if 'file' not in request.files:
                flash('No file part')
                return redirect(request.url)
            file = request.files['file']
            # if user does not select file, browser also
            # submit an empty part without filename
            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                return {"filename": filename}


    return app, api


app, api = custom_create_app()
