import logging
import os
from flask import Blueprint, jsonify, request
import requests
import time
from threading import Thread
import socket

upload_video_api = Blueprint('upload_video_api', __name__)

def get_ipv4_address():
    hostname = socket.gethostname()
    return socket.gethostbyname(hostname)

# Configure upload folder
UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

processing_status = {"is_complete": False}

# Background processing task
def process_data():
    global processing_status
    time.sleep(1)  # Simulate a 10-second processing time
    processing_status["is_complete"] = True

@upload_video_api.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video file in request"}), 400
    
    global processing_status

    video = request.files['video']
    title = video.filename
    email = request.form.get('email')

    if not video or video.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save the video to the upload folder
    video_path = os.path.join("./uploads", video.filename)
    video.save(video_path)

    start_frame_extraction(video.filename, email)
    start_processing()
    return jsonify({
        "message": "Video uploaded successfully",
        "video_path": video_path,
        "title": title,
        "email": email
    }), 200

def start_processing():
    global processing_status
    if not processing_status["is_complete"]:  # Avoid restarting if already complete
        processing_status["is_complete"] = False
        Thread(target=process_data).start()  # Start background task
        return jsonify({"message": "Processing started"}), 202
    return jsonify({"message": "Processing already completed"}), 400

@upload_video_api.route('/status', methods=['GET'])
def check_status():
    global processing_status
    return jsonify({"isComplete": processing_status["is_complete"]})


def start_frame_extraction(title, email):
    ipv4_address = get_ipv4_address()
    frame_extraction_url= f'http://{ipv4_address}:5000/api/frame_extraction_api/process_local_video'
    payload = {'title': title, 'email': email}
    headers = {'Content-Type': 'application/json'}

    try:
        response = requests.post(frame_extraction_url, json=payload, headers=headers, timeout=3000)
        response.raise_for_status()
        logging.debug("Received response from upload Video API.")
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"frame Extraction API request failed: {e}")
        return {'error': f"frame Extraction API request failed: {e}"}
    
