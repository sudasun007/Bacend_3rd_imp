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

# Initialize Flask app
main_model_api = Blueprint('main_model_api', __name__)
logging.basicConfig(level=logging.DEBUG)


# Mediapipe setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.5
)

# Predefined angles and mappings
'''class_angles = {
    'hachijidachijodanyoko': [
        [178.6704642, 121.4324436, 178.6704642, 121.4324436, 176.2882244, 176.2882244, 179.0472569, 155.9306886, 174.7958128, 0],
        [74.73670094, 179.6333059, 74.73670094, 179.6333059, 179.2610734, 179.2610734, 177.7071013, 157.4409467, 146.1440207, 0]
    ],
    'sanchindachiageuke': [
        [117.6311735, 177.122714, 117.6311735, 177.122714, 172.4470531, 172.4470531, 179.84531, 178.0124464, 154.3915227, 0],
        [88.35833, 130.8359801, 88.35833, 130.8359801, 178.8655222, 178.8655222, 179.2608471, 159.6653805, 172.8783815, 0]
    ],
    'sanchindachijodantsuki': [
        [79.67428505, 117.6143462, 79.67428505, 117.6143462, 175.4425771, 175.4425771, 179.9083308, 179.6819875, 162.7090183, 0],
        [132.2753167, 61.93098373, 132.2753167, 61.93098373, 177.3881788, 177.3881788, 177.6381035, 156.7447632, 179.4797426, 0]
    ],
    'sanchindachisotouke': [
        [11.47716376, 114.5169405, 11.47716376, 114.5169405, 176.3834111, 176.3834111, 175.592297, 179.2147185, 163.3744971, 0],
        [87.20938557, 16.70051146, 87.20938557, 16.70051146, 178.365065, 178.365065, 174.4879716, 162.4809264, 176.8140283, 0]
    ],
    'shikodachigedanbarai': [
        [176.4795284, 153.3314441, 176.4795284, 153.3314441, 107.3410318, 107.3410318, 97.62956052, 153.8734055, 156.7703905, 0],
        [147.1951634, 175.074177, 147.1951634, 175.074177, 103.3349196, 103.3349196, 128.5222031, 157.1361985, 145.8509965, 0]
    ],
    'sotoukemaegeri': [
        [141.6715664, 12.61353582, 141.6715664, 12.61353582, 177.0001987, 177.0001987, 175.727286, 164.0403825, 160.4981336, 0],
        [15.39189415, 84.51329129, 15.39189415, 84.51329129, 174.1109382, 174.1109382, 158.9654381, 178.2361718, 159.3976762, 0]
    ],
    'zenkutsudachiawasetsuki': [
        [115.8580351, 130.7511613, 115.8580351, 130.7511613, 171.9446945, 171.9446945, 172.6747746, 144.105041, 178.9400008, 0]
    ],
    'zenkutsudachichudantsuki': [
        [115.8580351, 130.7511613, 115.8580351, 130.7511613, 171.9446945, 171.9446945, 172.6747746, 144.105041, 178.9400008, 0],
        [156.0357435, 61.09435279, 156.0357435, 61.09435279, 172.3895637, 172.3895637, 162.5085944, 157.2766185, 159.0072517, 0]
    ],
    'zenkutsudachiempiuke': [
        [5.537744135, 138.5696075, 5.537744135, 138.5696075, 176.3914556, 176.3914556, 173.9572285, 166.2690671, 157.8046774, 0],
        [156.0357435, 61.09435279, 156.0357435, 61.09435279, 172.3895637, 172.3895637, 162.5085944, 157.2766185, 159.0072517, 0]
    ]
}'''

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
            logging.debug(f"Loading model from {model_path}")
            model = tf.keras.models.load_model(model_path)
            logging.info("Model loaded successfully.")
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

'''# Calculate angle function
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(cosine_angle))'''

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
    

'''# Extract angles function
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
        return []'''

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
        

'''# Compare angles function
def compare_angles(extracted_angles, predefined_angles):
    similarities = []
    for angles in predefined_angles:
        # Trim or pad extracted_angles to match predefined_angles length
        aligned_extracted_angles = extracted_angles[:len(angles)]
        if len(angles) < len(extracted_angles):
            aligned_extracted_angles += [0] * (len(angles) - len(aligned_extracted_angles))
        
        similarity = np.mean(1 - np.abs(np.array(aligned_extracted_angles) - np.array(angles)) / 180) * 100
        similarities.append(similarity)
    return max(similarities)'''

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


# Flask route for processing images
@main_model_api.route('/main_model', methods=['POST'])
def main_model():
    try:
        logging.info("Main model API endpoint called.")
        data = request.json
        

        preprocessed_frames = data.get('preprocessed_frames')
        video_name = data.get('video_name', 'video2.mp4')
        player_email = data.get('player_email', 'sandalisithumani@gmail.com')

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

            # Remaining processing logic for classification, similarity, and saving results
            img_array = preprocess_image(img)
            if img_array is None:
                logging.error(f"Skipping frame {i + 1}: Failed preprocessing.")
                continue

            logging.debug(f"Input shape for model.predict: {img_array.shape}")
            predictions = model.predict(img_array)
            predicted_class_index = np.argmax(predictions)
            class_name = class_index_to_name[predicted_class_index]

            #predefined_angles = class_angles.get(class_name, [])
            #similarity = compare_angles(extracted_angles, predefined_angles) if predefined_angles else None

            #if not comment the predefined_angles and similarity and use the below one
            similarity = compare_angles(extracted_angles, class_name)


            if similarity is not None:
                if r_id is None:
                    r_id = save_result(video_name, player_email)
                save_single_pose(class_name, similarity, r_id)
                single_pose_results.append(similarity)


        # Calculate final result
        final_result = np.mean(single_pose_results) if single_pose_results else 0
        logging.debug(f"Type of final_result: {type(final_result)}")
        update_result(r_id, final_result)

        return jsonify({'message': 'Processing complete', 'final_result': final_result}), 200
    except Exception as e:
        logging.error(f"Error in main model: {e}")
        return jsonify({'error': str(e)}), 501
    


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
    
    
def save_single_pose(pose_name, pose_result, r_id):
    try:
        db = get_db()
        cursor = db.cursor()
        cursor.execute('SELECT Pose_id FROM Correct_pose WHERE Correct_Pose_name = %s', (pose_name,))
        c_id = cursor.fetchone()
        
        if c_id is None:
            raise ValueError(f"Pose name {pose_name} not found in Correct_pose table")

        cursor.execute('''
            INSERT INTO Single_pose (Pose_name, Single_pose_result, R_id, C_id)
            VALUES (%s, %s, %s, %s)
        ''', (pose_name, float(pose_result), r_id, c_id[0]))  # Explicit conversion
        db.commit()
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

def get_image_filepath(single_pose_id):
    filename = f'{single_pose_id}.jpg'
    filepath = os.path.join(IMAGE_FOLDER, filename)
    return filepath

def delete_single_pose(single_pose_id):
    db = get_db()
    cursor = db.cursor()
    cursor.execute('DELETE FROM Single_pose WHERE Single_pose_id = %s', (single_pose_id,))
    db.commit()
    cursor.close()
