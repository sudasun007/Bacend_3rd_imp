from flask import Blueprint, jsonify, current_app, send_file, request
from http import HTTPStatus
from flasgger import swag_from

correct_pose_api = Blueprint('correct_pose_api', __name__)

@correct_pose_api.route('/correctposes', methods=['GET'])
@swag_from({
    'responses': {
        HTTPStatus.OK.value: {
            'description': 'List of Correct Poses',
            'schema': {
                'type': 'array',
                'items': {
                    'type': 'object',
                    'properties': {
                        'Pose_id': {'type': 'integer'},
                        'Correct_Pose_name': {'type': 'string'},
                        'Pose_Details': {'type': 'string'},
                        'Img_link': {'type': 'string'}
                    }
                }
            }
        }
    }
})
def get_correct_poses():
    cursor = current_app.mysql.cursor(dictionary=True)
    cursor.execute("SELECT Pose_id, Correct_Pose_name, Pose_Details, Img_link FROM Correct_pose")
    correct_poses = cursor.fetchall()
    cursor.close()
    
    for pose in correct_poses:
        # Assuming Img_link is the Google Drive link
        direct_link = pose['Img_link'].replace('uc?id=', 'uc?export=view&id=')
        pose['Img_link'] = direct_link
    
    return jsonify(correct_poses), HTTPStatus.OK

@correct_pose_api.route('/correctposes/<pose_name>', methods=['GET'])
def get_correct_pose(pose_name):
    cursor = current_app.mysql.cursor(dictionary=True)
    cursor.execute("SELECT Pose_id, Correct_Pose_name, Pose_Details, Img_link FROM Correct_pose WHERE Correct_Pose_name = %s", (pose_name,))
    correct_pose = cursor.fetchone()
    cursor.close()

    if not correct_pose:
        return jsonify({'error': 'Pose not found'}), HTTPStatus.NOT_FOUND
    
    # Assuming Img_link is the Google Drive link
    direct_link = correct_pose['Img_link'].replace('uc?id=', 'uc?export=view&id=')
    correct_pose['Img_link'] = direct_link
    
    return jsonify(correct_pose), HTTPStatus.OK

@correct_pose_api.route('/images/<pose_id>', methods=['GET'])
def get_pose_image(pose_id):
    cursor = current_app.mysql.cursor(dictionary=True)
    cursor.execute("SELECT Img_link FROM Correct_pose WHERE Pose_id = %s", (pose_id,))
    img_link = cursor.fetchone()['Img_link']
    cursor.close()

    # Assuming Img_link is the Google Drive link
    direct_link = img_link.replace('uc?id=', 'uc?export=view&id=')

    # Return the image as a file
    return send_file(direct_link, mimetype='image/jpeg')  # Adjust mimetype as per your image type
