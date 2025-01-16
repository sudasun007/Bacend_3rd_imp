from flask import Flask, request, jsonify, Blueprint
import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import base64
import time
from db_config import get_db, close_db
import logging
import os
import json
import socket

def get_ipv4_address():
    hostname = socket.gethostname()
    return socket.gethostbyname(hostname)

# Initialize Flask app
main_model_api = Blueprint('main_model_api', __name__)
logging.basicConfig(level=logging.DEBUG)


IMAGE_FOLDER = 'C:/Users/USER/Desktop/New_Flask_Backend/local_pose_images'

# Ensure the folder exists
if not os.path.exists(IMAGE_FOLDER):
    os.makedirs(IMAGE_FOLDER)



# Mediapipe setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.5
)


class_angles = {
    'hachijidachijodanyoko': [
        [89.04714, 65.86339, 90.93809, 138.40607, 166.8372, 146.91685, 130.31555, 122.15627, 20.859364, 26.549637],
        [36.5793, 99.51401, 117.69932, 127.01856, 170.74667, 166.65123, 141.54074, 135.26541, 17.554852, 20.548319]
    ],
    'sanchindachiageuke': [
        [126.96851, 81.76038, 152.83405, 164.39342, 145.93587, 172.3908, 123.81934, 134.86362, 27.332039, 16.215887],
        [42.162388, 116.37471, 146.77101, 149.65286, 179.83214, 167.32835, 104.20513, 128.25916, 29.190786, 22.458845]
    ],
    'sanchindachijodantsuki': [
        [107.85367, 21.094286, 162.01074, 107.7524, 147.16394, 146.6018, 152.80084, 158.29625, 13.7039995, 9.376472],
        [54.350147, 105.212845, 143.42456, 160.90521, 167.0334, 149.70805, 104.98921, 123.37991, 32.072716, 25.107214]
    ],
    'sanchindachisotouke': [
        [89.46901, 71.801476, 149.67778, 152.87282, 141.6359, 147.75536, 144.54117, 153.37794, 18.27299, 9.927412],
        [51.96784, 92.94476, 132.67818, 147.39087, 156.49437, 146.5574, 137.1467, 127.166985, 16.040712, 25.639128]
    ],
    'shikodachigedanbarai': [
        [62.581818, 19.325127, 113.50904, 171.34323, 32.088352, 39.008385, 40.004086, 31.109266, 81.80888, 78.982864],
        [115.83695, 55.40469, 153.50008, 120.65104, 40.60882, 46.726242, 40.587418, 48.01508, 74.96912, 63.343178]
    ],
    'sotoukemaegeri': [
        [66.401024, 78.91096, 107.96682, 140.094, 104.378105, 172.18393, 154.61859, 171.18639, 19.1304, 4.6232147],
        [68.65641, 0.6974545, 132.9251, 110.71054, 166.6013, 111.18552, 170.86172, 137.88292, 4.4507823, 31.32719]
    ],
    'zenkutsudachiawasetsuki': [
        [62.699413, 118.50581, 150.93295, 123.81974, 127.30449, 129.80608, 86.64032, 136.8235, 39.998837, 19.8054],
        [76.51966, 63.037014, 169.2019, 173.23474, 161.1424, 126.62961, 141.07997, 118.55954, 16.919363, 39.934387]
    ],
    'zenkutsudachichudantsuki': [
        [119.33198, 28.959768, 167.27402, 113.7191, 163.23285, 99.924805, 157.8609, 84.51811, 7.045228, 70.86057]
    ],
    'zenkutsudachiempiuke': [
        [116.84017, 71.0987, 162.62036, 171.38647, 95.296616, 120.855995, 134.72871, 167.42279, 27.158785, 4.8582916],
        [66.50024, 109.27068, 168.59831, 162.92047, 159.45186, 107.5432, 160.3982, 140.94868, 9.020206, 20.688852]
    ]
}

# Mapping from class indexes to class names
class_index_to_name = {
    0: 'hachijidachijodanyoko',
    1: 'sanchindachiageuke',
    2: 'sanchindachijodantsuki',
    3: 'sanchindachisotouke',
    4: 'shikodachigedanbarai',
    5: 'sotoukemaegeri',
    6: 'zenkutsudachiawasetsuki',
    7: 'zenkutsudachichudantsuki',
    8: 'zenkutsudachiempiuke'
}

@main_model_api.before_app_request
def tf_load_model():
    global model
    if 'model' not in globals() or model is None:
        try:
            model_path = 'models_files/cnn_model_new19_2.keras'
            logging.debug(f"Loading  main model from {model_path}")
            model = tf.keras.models.load_model(model_path)
            logging.info("Main Model loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            model = None

# Extract keypoints function
def extract_keypoints(image):
    
    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if not results.pose_landmarks:
            return None
        keypoints = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark], dtype=np.float32)
        
    return keypoints

# Combined normalization function
def combined_normalization(landmarks):
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
    hip_center = np.array([(left_hip[0] + right_hip[0]) / 2, 
                           (left_hip[1] + right_hip[1]) / 2, 
                           (left_hip[2] + right_hip[2]) / 2])
    translated_landmarks = [np.array([lm[0], lm[1], lm[2]]) - hip_center for lm in landmarks]
    hip_width = np.linalg.norm(np.array(left_hip) - np.array(right_hip))
    scaled_landmarks = [coords / hip_width for coords in translated_landmarks]
    return np.array(scaled_landmarks, dtype=np.float32)

# Preprocess keypoints function
def preprocess_keypoints(keypoints):
    reshaped = keypoints.reshape(-1, 33, 3)
    reshaped_with_channel = np.expand_dims(reshaped, axis=-1)
    return reshaped_with_channel

# Classify pose function
def classify_pose(model, keypoints):
    reshaped_keypoints = preprocess_keypoints(keypoints)
    prediction = model.predict(reshaped_keypoints)
    return np.argmax(prediction)

# updated Calculate angle function
# Calculate angles
def calculate_angle(point1, point2, point3):
    v1 = np.array(point1) - np.array(point2)
    v2 = np.array(point3) - np.array(point2)
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    if v1_norm == 0 or v2_norm == 0:
        return 0.0
    dot_product = np.dot(v1, v2)
    angle = np.arccos(dot_product / (v1_norm * v2_norm))
    return np.degrees(angle)
    

#updated extract angles functon
def extract_angles(keypoints):
    try:
        mp_landmarks = mp.solutions.pose.PoseLandmark
        angles = [
            calculate_angle(keypoints[mp_landmarks.LEFT_ELBOW.value], keypoints[mp_landmarks.LEFT_SHOULDER.value], keypoints[mp_landmarks.LEFT_HIP.value]),
            calculate_angle(keypoints[mp_landmarks.RIGHT_HIP.value], keypoints[mp_landmarks.RIGHT_SHOULDER.value], keypoints[mp_landmarks.RIGHT_ELBOW.value]),
            calculate_angle(keypoints[mp_landmarks.LEFT_SHOULDER.value], keypoints[mp_landmarks.LEFT_ELBOW.value], keypoints[mp_landmarks.LEFT_WRIST.value]),
            calculate_angle(keypoints[mp_landmarks.RIGHT_WRIST.value], keypoints[mp_landmarks.RIGHT_ELBOW.value], keypoints[mp_landmarks.RIGHT_SHOULDER.value]),
            calculate_angle(keypoints[mp_landmarks.LEFT_KNEE.value], keypoints[mp_landmarks.LEFT_HIP.value], keypoints[mp_landmarks.LEFT_SHOULDER.value]),
            calculate_angle(keypoints[mp_landmarks.RIGHT_SHOULDER.value], keypoints[mp_landmarks.RIGHT_HIP.value], keypoints[mp_landmarks.RIGHT_KNEE.value]),
            calculate_angle(keypoints[mp_landmarks.LEFT_HIP.value], keypoints[mp_landmarks.LEFT_KNEE.value], keypoints[mp_landmarks.LEFT_ANKLE.value]),
            calculate_angle(keypoints[mp_landmarks.RIGHT_ANKLE.value], keypoints[mp_landmarks.RIGHT_KNEE.value], keypoints[mp_landmarks.RIGHT_HIP.value]),
            calculate_angle(keypoints[mp_landmarks.LEFT_KNEE.value], keypoints[mp_landmarks.LEFT_ANKLE.value], keypoints[mp_landmarks.LEFT_HIP.value]),
            calculate_angle(keypoints[mp_landmarks.RIGHT_HIP.value], keypoints[mp_landmarks.RIGHT_ANKLE.value], keypoints[mp_landmarks.RIGHT_KNEE.value])
        ]
        return angles
    except Exception as e:
        logging.error(f"Error extracting angles: {e}")
        return []
        

# compae angles update function
# Compare angles with predefined data
def compare_angles(extracted_angles, class_name):
    predefined_angles = class_angles.get(class_name, [])
    if not predefined_angles:
        return 0.0
    similarities = []
    for predefined in predefined_angles:
        diff = np.abs(np.array(predefined) - np.array(extracted_angles))
        similarities.append(100 - np.mean(diff))
        mae = np.mean(diff)
        print(mae)
    return max(similarities)
    

# Load image function
def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image could not be loaded!")
    return image

def preprocess_image(image):
    try:
        keypoints = extract_keypoints(image)
        if keypoints is None:
            return None
        normalized_keypoints = combined_normalization(keypoints)
        return preprocess_keypoints(normalized_keypoints)
    except Exception as e:
        logging.error(f"Error during preprocessing: {e}")
        return None


'''# Flask route for processing images
@main_model_api.route('/main_model', methods=['POST'])
def main_model():
    try:
        logging.info("Main model API endpoint called.")
        data = request.json
        

        preprocessed_frames = data.get('preprocessed_frames')
        video_name = data.get('video_name', 'video2.mp4')
        player_email = data.get('email', 'sandalisithumani@gmail.com')

        if not preprocessed_frames:
            logging.error("Missing required preprocessed frames.")
            return jsonify({'error': 'Missing preprocessed frames'}), 400

        if model is None:
            logging.error("Model is not loaded. Ensure the model file exists and is accessible.")
            return jsonify({'error': 'Model not loaded. Contact the administrator.'}), 500

        # Process each frame
        decoded_images = []
        for i, frame in enumerate(preprocessed_frames):
            logging.debug(f"Decoding frame {i + 1}/{len(preprocessed_frames)}.")
            img_data = base64.b64decode(frame)
            np_img = np.frombuffer(img_data, dtype=np.uint8)
            img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
            decoded_images.append(img)

        logging.info(f"Decoded {len(decoded_images)} frames successfully.")
        single_pose_results = []
        r_id = None
        for i, img in enumerate(decoded_images):
            logging.debug(f"Processing image {i + 1}/{len(decoded_images)}.")
            keypoints = extract_keypoints(img)

            if keypoints is None:  # Handle case when keypoints are not detected
                logging.warning(f"Frame {i + 1} could not detect landmarks. Skipping.")
                continue

            extracted_angles = extract_angles(keypoints)
            if not extracted_angles:
                logging.warning(f"Frame {i + 1} could not extract angles. Skipping.")
                continue

            # Preprocess the image
            img_array = preprocess_image(img)
            if img_array is None:
                logging.error(f"Skipping frame {i + 1}: Failed preprocessing.")
                continue

            logging.debug(f"Input shape for model.predict: {img_array.shape}")
            predictions = model.predict(img_array)
            predicted_class_index = np.argmax(predictions)
            class_name = class_index_to_name[predicted_class_index]

            similarity = compare_angles(extracted_angles, class_name)

            if similarity is not None:
                if r_id is None:
                    r_id = save_result(video_name, player_email)

                # Convert the image back to a writable format for saving
                _, img_encoded = cv2.imencode('.jpg', img)
                image_data = img_encoded.tobytes()

                save_single_pose(class_name, similarity, r_id, image_data, i)
                single_pose_results.append(similarity)

                 # Calculate final result
        final_result = np.mean(single_pose_results) if single_pose_results else 0
        logging.debug(f"Type of final_result: {type(final_result)}")
        update_result(r_id, final_result)

        return jsonify({'message': 'Processing complete', 'final_result': final_result}), 200
    except Exception as e:
        logging.error(f"Error in main model: {e}")
        return jsonify({'error': str(e)}), 501'''


@main_model_api.route('/main_model', methods=['POST'])
def main_model():
    try:
        logging.info("Main model API endpoint called.")
        data = request.json

        # Extract email and video name
        preprocessed_frames = data.get('preprocessed_frames')
        video_name = data.get('video_name', 'default_video.mp4')
        player_email = data.get('email')

        if not preprocessed_frames or not player_email:
            logging.error("Missing required data.")
            return jsonify({'error': 'Missing preprocessed frames or email'}), 400

        if model is None:
            logging.error("Model is not loaded. Ensure the model file exists and is accessible.")
            return jsonify({'error': 'Model not loaded. Contact the administrator.'}), 500

        # Save the result and get the result ID
        r_id = save_result(video_name, player_email)

        # Process the frames (existing logic)
        decoded_images = []
        for frame in preprocessed_frames:
            img_data = base64.b64decode(frame)
            np_img = np.frombuffer(img_data, dtype=np.uint8)
            img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
            decoded_images.append(img)

        logging.info(f"Decoded {len(decoded_images)} frames successfully.")
        single_pose_results = []

        for img in decoded_images:
            keypoints = extract_keypoints(img)
            if keypoints is None:
                continue

            extracted_angles = extract_angles(keypoints)
            img_array = preprocess_image(img)
            predictions = model.predict(img_array)
            predicted_class_index = np.argmax(predictions)
            class_name = class_index_to_name[predicted_class_index]
            similarity = compare_angles(extracted_angles, class_name)

            if similarity is not None:
                _, img_encoded = cv2.imencode('.jpg', img)
                image_data = img_encoded.tobytes()
                save_single_pose(class_name, similarity, r_id, image_data, 0)
                single_pose_results.append(similarity)

        # Update the final result
        final_result = np.mean(single_pose_results) if single_pose_results else 0
        update_result(r_id, final_result)

        logging.info(f"Generated result ID: {r_id}")

        return jsonify({
            'message': 'Processing complete',
            'result_id': r_id,
            'final_result': final_result
        }), 200

    except Exception as e:
        logging.error(f"Error in main_model: {e}")
        return jsonify({'error': str(e)}), 500



def save_result(video_name, player_email):
    try:
        db = get_db()  # Get the database connection
        cursor = db.cursor()
        cursor.execute(''' 
            INSERT INTO Result (Video_name, Date, Final_result, Rank_P, P_email)
            VALUES (%s, %s, %s, %s, %s)''', 
            (video_name, time.strftime('%Y-%m-%d'), float(0), 'Not Ranked', player_email))
        db.commit()
        result_id = cursor.lastrowid
        cursor.close()
        return result_id
    except Exception as e:
        db.rollback()
        raise e
    
    
def save_single_pose(pose_name, pose_result, r_id, image_data, frame_index):
    try:
        db = get_db()
        cursor = db.cursor()
        cursor.execute('SELECT Pose_id FROM Correct_pose WHERE Correct_Pose_name = %s', (pose_name,))
        c_id = cursor.fetchone()

        if c_id is None:
            raise ValueError(f"Pose name {pose_name} not found in Correct_pose table")

        # Insert the Single_pose record
        cursor.execute('''
            INSERT INTO Single_pose (Pose_name, Single_pose_result, R_id, C_id)
            VALUES (%s, %s, %s, %s)
        ''', (pose_name, float(pose_result), r_id, c_id[0]))
        db.commit()

        single_pose_id = cursor.lastrowid  # Get the generated Single_pose_id

        # Save the image locally using Single_pose_id
        image_filename = f"{single_pose_id}.jpg"
        image_path = os.path.join(IMAGE_FOLDER, image_filename)
        with open(image_path, 'wb') as img_file:
            img_file.write(image_data)

        cursor.close()
    except Exception as e:
        db.rollback()
        raise e

def update_result(r_id, final_result):
    try:
        db = get_db()
        cursor = db.cursor()
        cursor.execute('''
            UPDATE Result
            SET Final_result = %s
            WHERE Result_id = %s
        ''', (float(final_result), r_id))  # Explicit conversion
        db.commit()
        cursor.close()
    except Exception as e:
        db.rollback()
        raise e
    
def calculate_and_update_ranks():
    try:
        db = get_db()  # Get the database connection
        cursor = db.cursor()

        # Step 1: Retrieve all results sorted by final_result in descending order
        cursor.execute('''
            SELECT Result_id, P_email, Final_result FROM Result
            ORDER BY Final_result DESC
        ''')
        results = cursor.fetchall()

        # Step 2: Assign ranks and update the database
        rank = 1
        previous_final_result = None  # To track previous Final_result for tie handling
        tie_count = 0  # To count the number of players with the same Final_result

        for i, result in enumerate(results):
            result_id = result[0]
            player_email = result[1]
            final_result = result[2]

            # Step 3: Check if the current result is the same as the previous one (tie situation)
            if final_result == previous_final_result:
                tie_count += 1  # Increment tie count for the same final_result
            else:
                # If it's not a tie, update the rank to the correct rank based on position
                rank = i + 1  # The rank is based on the position in the sorted list
                tie_count = 0  # Reset tie count for new final_result

            # Step 4: Update the player's rank in the database
            cursor.execute('''
                UPDATE Result
                SET Rank_P = %s
                WHERE Result_id = %s
            ''', (rank, result_id))
            
            previous_final_result = final_result  # Update the previous result

        db.commit()
        cursor.close()
        logging.info("Ranks updated successfully.")
    
    except Exception as e:
        db.rollback()
        logging.error(f"Error updating ranks: {e}")


@main_model_api.route('/delete_image', methods=['POST'])
def delete_image():
    single_pose_id = request.json.get('single_pose_id')
    if not single_pose_id:
        return jsonify({'error': 'No single_pose_id provided'}), 400

    try:
        filepath = get_image_filepath(single_pose_id)
        if not filepath:
            return jsonify({'error': 'Image not found'}), 404

        if os.path.exists(filepath):
            os.remove(filepath)
            delete_single_pose(single_pose_id)
            return jsonify({'message': 'Image deleted successfully'}), 200
        else:
            return jsonify({'error': 'Image file not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500



@main_model_api.route('/get_single_pose_details', methods=['GET'])
def get_single_pose_details():
    try:
        logging.debug("get_single_pose_details endpoint called.")
        logging.debug(f"Request args: {request.args}")

        # Retrieve result_id
        result_id = request.args.get('result_id')
        logging.debug(f"Received result_id: {result_id}")

        if not result_id:
            return jsonify({'error': 'Result ID is required'}), 400

        db = get_db()
        cursor = db.cursor()

        # Fetch pose details
        query = '''
            SELECT 
                sp.Pose_name AS Single_pose_name,
                sp.Single_pose_result,
                cp.Correct_Pose_name,
                cp.Pose_Details,
                cp.Img_link,
                sp.Single_pose_id
            FROM Single_pose sp
            JOIN Correct_pose cp ON sp.C_id = cp.Pose_id
            WHERE sp.R_id = %s
        '''
        cursor.execute(query, (result_id,))
        poses = cursor.fetchall()

        if not poses:
            logging.debug(f"No pose details found for result_id: {result_id}")
            return jsonify({'error': 'No pose details found for the given result ID'}), 404

        response = []
        for pose in poses:
            single_pose_id = pose[5]
            single_pose_image_path = os.path.join(IMAGE_FOLDER, f'{single_pose_id}.jpg')
            single_pose_image_url = None

            if os.path.exists(single_pose_image_path):
                ipv4_address = get_ipv4_address()
                single_pose_image_url = f"http://{ipv4_address}:5000/local_pose_images/{single_pose_id}.jpg"

            response.append({
                "Single_pose_name": pose[0],
                "Single_pose_result": float(pose[1]),
                "Correct_Pose_name": pose[2],
                "Pose_Details": pose[3],
                "Correct_Pose_Img_link": pose[4],
                "Single_Pose_Img_link": single_pose_image_url or "Image not found"
            })

        cursor.close()
        return jsonify(response), 200

    except Exception as e:
        logging.error(f"Error fetching single pose details: {e}")
        return jsonify({'error': str(e)}), 500


def get_single_leaderboard(player_email):
    try:
        db = get_db()  # Get the database connection
        cursor = db.cursor()
        cursor.execute('''
            SELECT Video_name, Final_result
            FROM Result
            WHERE P_email = %s
            ORDER BY Date DESC
        ''', (player_email,))
        
        results = cursor.fetchall()
        cursor.close()
        
        # Format the results as a list of dictionaries with video name and email
        results_list = []
        for result in results:
            results_list.append({
                'video_name': result[0],
                'final_score': float(result[1])
            })
        return results_list

    except Exception as e:
        logging.error(f"Error retrieving player results: {e}")
        return None
        

@main_model_api.route('/single_leaderboard', methods=['POST'])
def single_leaderboard():
    try:
        player_email = request.get_json().get('Email')

        if not player_email:
            return jsonify({'error': 'Player email is required'}), 400

        results = get_single_leaderboard(player_email)

        if results is None:
            return jsonify({'error': 'Error retrieving player results'}), 500

        return jsonify(results), 200

    except Exception as e:
        logging.error(f"Error in single_leaderboard endpoint: {e}")
        return jsonify({'error': str(e)}), 500
    
def get_global_leaderboard():
    try:
        db = get_db()  # Get the database connection
        cursor = db.cursor()
        cursor.execute('''
            SELECT F_name, L_name, Final_result
            FROM Result
            JOIN Player ON Result.P_email = Player.Email
            ORDER BY Final_result DESC
        ''')
        
        results = cursor.fetchall()
        cursor.close()
        
        # Format the results as a list of dictionaries
        leaderboard = []
        for result in results:
            leaderboard.append({
                
                'first_name': result[0],
                'last_name': result[1],
                'final_score': float(result[2])
            })

        return leaderboard

    except Exception as e:
        logging.error(f"Error retrieving global leaderboard: {e}")
        return None

@main_model_api.route('/global_leaderboard', methods=['GET'])
def global_leaderboard():
    try:
        leaderboard = get_global_leaderboard()

        if leaderboard is None:
            return jsonify({'error': 'Error retrieving global leaderboard'}), 500

        return jsonify(leaderboard), 200

    except Exception as e:
        logging.error(f"Error in global_leaderboard endpoint: {e}")
        return jsonify({'error': str(e)}), 500
    

@main_model_api.route('/get_user_rank', methods=['GET'])
def get_user_rank():
    player_email = request.args.get('player_email')
    if not player_email:
        return jsonify({'error': 'Player email is required'}), 400

    try:
        # Optionally recalculate ranks
        calculate_and_update_ranks()

        # Fetch the user's rank
        db = get_db()
        cursor = db.cursor()
        cursor.execute('''
            SELECT Rank_P FROM Result WHERE P_email = %s
        ''', (player_email,))
        result = cursor.fetchone()
        cursor.close()

        if result is None:
            return jsonify({'error': 'Rank not found'}), 404

        rank = result[0]
        return jsonify({'rank': rank}), 200
    except Exception as e:
        logging.error(f"Error fetching user rank: {e}")
        return jsonify({'error': str(e)}), 500
