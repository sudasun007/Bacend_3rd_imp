from flask import Flask
from flasgger import Swagger
from api.route.home import home_api
from api.route.correct_pose import correct_pose_api
from api.route.upload_video import upload_video_api
from api.route.edit_video import edit_video_api
from api.route.frame_extraction_comparison import frame_extraction_api
from api.route.preprocessing import preprocessing_api
from api.route.main_model import main_model_api
from api.route.guide_video import guide_video_api
from api.route.profile_routes import profile_bp
from flask_cors import CORS


import mysql.connector

def create_app():
    app = Flask(__name__)
    CORS(app)
      # Replace with a random secret key for production use
    app.secret_key = '1234'

    app.config['SWAGGER'] = {
        'title': 'Flask API Starter Kit',
    }
    swagger = Swagger(app)
    
    ## Initialize Config
    app.config.from_pyfile('config.py')
    
    ## Initialize MySQL connection
    app.mysql = mysql.connector.connect(
        host=app.config['MYSQL_DATABASE_HOST'],
        user=app.config['MYSQL_DATABASE_USER'],
        password=app.config['MYSQL_DATABASE_PASSWORD'],
        database=app.config['MYSQL_DATABASE_DB']
    )
    
    # Register blueprints
    app.register_blueprint(home_api, url_prefix='/api')
    app.register_blueprint(correct_pose_api, url_prefix='/api')
    app.register_blueprint(upload_video_api, url_prefix='/api')
    app.register_blueprint(edit_video_api, url_prefix='/api')
    app.register_blueprint(frame_extraction_api, url_prefix='/api/frame_extraction_api')
    app.register_blueprint(main_model_api, url_prefix='/api/main_model_api')
    app.register_blueprint(preprocessing_api, url_prefix='/api/preprocessing_api')
    app.register_blueprint(guide_video_api, url_prefix='/api')
    app.register_blueprint(profile_bp, url_prefix='/api')

    # Set up session interface (optional for customization)
    from flask_session import Session
    app.config['SESSION_TYPE'] = 'filesystem'
    Session(app)


    return app

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-p', '--port', default=5000, type=int, help='port to listen on')
    args = parser.parse_args()
    port = args.port

    app = create_app()

    app.run(host='0.0.0.0', port=5000)
