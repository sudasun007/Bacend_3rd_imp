from flask import Blueprint, request, jsonify, current_app
import mysql.connector

guide_video_api = Blueprint('guide_video_api', __name__)

@guide_video_api.route('/all_guide_videos', methods=['GET'])
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
