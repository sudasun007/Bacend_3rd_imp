from flask import Blueprint, jsonify, request
import requests
import logging
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import json
import cv2
from ultralytics import YOLO

# Initialize Blueprint and Logging
preprocessing_api = Blueprint('preprocessing_api', __name__)
logging.basicConfig(level=logging.DEBUG)

# Load YOLOv8 model
try:
    model_path = 'models_files/yolov8n-seg.pt'  # Update with your actual model path
    logging.info(f"Loading YOLOv8 model from {model_path}...")
    model = YOLO(model_path)
    logging.info("YOLOv8 model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load YOLOv8 model: {e}")
    raise

# Function to preprocess frames
def preprocess_frames(raw_frames):
    """
    Preprocess the frames by decoding raw image data, resizing, enhancing, and background removal.
    Args:
        raw_frames (list): List of raw image data (Base64 strings or binary data).
    Returns:
        np.array: Array of preprocessed frames.
    """
    preprocessed_frames = []
    logging.info("Starting preprocessing of frames...")

    for i, raw_frame in enumerate(raw_frames):
        try:
            logging.debug(f"Preprocessing frame {i + 1}/{len(raw_frames)}...")

            # Decode the Base64 string into an image
            image_data = base64.b64decode(raw_frame)
            image = Image.open(BytesIO(image_data)).convert("RGB")
            image = np.array(image)

            # Enhance image using CLAHE
            logging.debug(f"Enhancing frame {i + 1} using CLAHE...")
            enhanced_image = enhance_image_with_clahe(image)
            

            # Apply YOLOv8 background removal
            logging.debug(f"Removing background from frame {i + 1} using YOLOv8...")
            background_removed_image = apply_yolov8_background_removal(enhanced_image)

            # Resize the image and normalize pixel values
            image_resized = cv2.resize(background_removed_image, (384, 512))
            logging.info("Resized 384*512 Successfully...")
            preprocessed_frames.append(image_resized)
        except Exception as e:
            logging.error(f"Error preprocessing frame {i + 1}: {e}")
            raise

    logging.info(f"Preprocessed {len(preprocessed_frames)} frames successfully.")
    return np.array(preprocessed_frames)

def enhance_image_with_clahe(image):
    """
    Enhance image contrast using CLAHE.
    Args:
        image (np.array): Input image in BGR format.
    Returns:
        np.array: Enhanced image.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    enhanced_lab = cv2.merge((cl, a, b))
    enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    return enhanced_image

def apply_yolov8_background_removal(image):
    """
    Remove background from image using YOLOv8.
    Args:
        image (np.array): Input image.
    Returns:
        np.array: Image with the background removed.
    """
    # Assuming 'model' is preloaded with YOLOv8 model
    results = model.predict(image)

    if not results or results[0].masks.data is None or len(results[0].masks.data) == 0:
        logging.warning("No mask detected by YOLOv8. Returning original image.")
        return image


    mask = results[0].masks.data[0].cpu().numpy()
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    mask = (mask * 255).astype(np.uint8)

    foreground = cv2.bitwise_and(image, image, mask=mask)
    background_removed = np.full_like(image, 255)  # White background
    background_removed[mask == 255] = foreground[mask == 255]

    return background_removed

@preprocessing_api.route('/preprocessing', methods=['POST'])
def preprocessing():
    logging.info("Preprocessing API endpoint called.")
    try:
        # Parse JSON payload
        data = request.get_json()
        logging.debug(f"Received data: {json.dumps(data, indent=2)}")

        raw_frames = data.get('frames')
        video_name = data.get('video_name', 'default_video_name.mp4')
        player_email = data.get('player_email', 'test@example.com')

        if not raw_frames:
            logging.error("No frames provided in the request.")
            return jsonify({'error': 'No frames provided'}), 400

        logging.debug(f"Video name: {video_name}, Player email: {player_email}")

        # Preprocess the frames
        preprocessed_frames = preprocess_frames(raw_frames)

        # Convert preprocessed frames to base64 strings
        preprocessed_frames_base64 = []
        for frame in preprocessed_frames:
            image = Image.fromarray(frame.astype(np.uint8))
            buffer = BytesIO()
            image.save(buffer, format="JPEG")
            frame_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            preprocessed_frames_base64.append(frame_base64)

        main_model_payload = {
            'preprocessed_frames': preprocessed_frames_base64
        }

        main_model_url = 'http://localhost:5000/api/main_model_api/main_model'
        response = requests.post(main_model_url, json=main_model_payload, timeout=300)
        response.raise_for_status()

        return jsonify({
            'message': 'Preprocessing completed successfully',
        }), 200

    except requests.exceptions.RequestException as req_err:
        logging.error(f"Request to main model API failed: {req_err}")
        return jsonify({'error': 'Failed to contact main model API', 'details':str(req_err)}),502

    except Exception as e:
        logging.error(f"Preprocessing failed: {e}")
        return jsonify({'error': 'Failed to preprocess frames', 'details':str(e)}),500