from flask import Blueprint, request, jsonify, current_app
import mysql.connector

guide_video_api = Blueprint('guide_video_api', __name__)

'''@guide_video_api.route('/all_guide_videos', methods=['GET'])
def get_all_guide_videos():
    try:
        connection = current_app.mysql
        cursor = connection.cursor(dictionary=True)

        query = "SELECT * FROM Guide_video"
        cursor.execute(query)
        
        results = cursor.fetchall()
        return jsonify(results), 200

    except mysql.connector.Error as error:
        return jsonify({'error': str(error)}), 500

    finally:
        if connection.is_connected():
            cursor.close()'''

@guide_video_api.route('/all_guide_videos', methods=['GET'])
def get_all_guide_videos():
    try:
        connection = current_app.mysql
        cursor = connection.cursor(dictionary=True)

        # Select only the name and link columns from the Guide_video table
        query = "SELECT Guide_video_name, Video_link, Thumbnail_link FROM Guide_video"
        cursor.execute(query)
        
        results = cursor.fetchall()
        return jsonify(results), 200

    except mysql.connector.Error as error:
        return jsonify({'error': str(error)}), 500

    finally:
        if connection.is_connected():
            cursor.close()


@guide_video_api.route('/search_guide_videos', methods=['GET'])
def search_guide_videos():
    query = request.args.get('query', '')

    if not query:
        return jsonify({'error': 'Query parameter is required'}), 400

    try:
        connection = current_app.mysql
        cursor = connection.cursor(dictionary=True)

        search_query = "SELECT * FROM Guide_video WHERE Guide_video_name LIKE %s"
        cursor.execute(search_query, ('%' + query + '%',))

        results = cursor.fetchall()
        return jsonify(results), 200

    except mysql.connector.Error as error:
        return jsonify({'error': str(error)}), 500

    finally:
        if connection.is_connected():
            cursor.close()

# Endpoint to log a guide video as watched
@guide_video_api.route('/watch_guide_video', methods=['POST'])
def watch_guide_video():
    data = request.get_json()
    player_email = data.get('email')
    guide_video_id = data.get('guide_video_id')
    if not player_email or not guide_video_id:
        return jsonify({'error': 'Both email and guide_video_id are required'}), 400
    try:
        connection = current_app.mysql
        cursor = connection.cursor()
        query = """
            INSERT INTO Player_Guide_video (P_Email, G_video_id) 
            VALUES (%s, %s) 
            ON DUPLICATE KEY UPDATE G_video_id = G_video_id
        """
        cursor.execute(query, (player_email, guide_video_id))
        connection.commit()
        return jsonify({'message': 'Guide video successfully logged.'}), 200
    except mysql.connector.Error as error:
        Login.info(str(error))
        return jsonify({'error': str(error)}), 500
    finally:
        if connection.is_connected():
            cursor.close()