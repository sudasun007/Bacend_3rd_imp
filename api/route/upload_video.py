from flask import Blueprint, jsonify, request

upload_video_api = Blueprint('upload_video_api', __name__)

@upload_video_api.route('/upload_video', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400
    
    # Handle the video upload process here
    # Example: You can perform any validation or initial processing
    video_name = file.filename
    # Example: Get video size for validation or processing
    video_size = len(file.read())
    
    # Example: Prepare response or pass on to next step
    return jsonify({'message': f'Video {video_name} uploaded successfully', 'size': video_size}), 200

# No need to run the blueprint directly, it will be run when registered in the main Flask app
