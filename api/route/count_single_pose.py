from flask import Blueprint, request, jsonify
import mysql.connector
import os
from dotenv import load_dotenv

# Create a new blueprint for the count single pose API
count_single_pose_api = Blueprint('count_single_pose_api', __name__)

# Initialize MySQL connection (use your app's configuration or directly connect)
def get_mysql_connection():
    connection = mysql.connector.connect(
        host=os.getenv('MYSQL_DATABASE_HOST'),
        user=os.getenv('MYSQL_DATABASE_USER'),
        password=os.getenv('MYSQL_DATABASE_PASSWORD'),
        database=os.getenv('MYSQL_DATABASE_DB')
    )
    return connection

@count_single_pose_api.route('/count_single_pose', methods=['POST'])
def count_single_pose():
    # Get R_id from the request body
    data = request.get_json()
    r_id = data.get('R_id')

    if not r_id:
        return jsonify({'error': 'R_id is required'}), 400

    # Get the MySQL connection and create a cursor
    connection = get_mysql_connection()
    cursor = connection.cursor(dictionary=True)
    
    # Query to count records in Single_pose where R_id matches
    query = "SELECT COUNT(*) AS count FROM Single_pose WHERE R_id = %s"
    cursor.execute(query, (r_id,))
    result = cursor.fetchone()

    # Close the cursor and connection
    cursor.close()
    connection.close()

    # Return the count, or 0 if no records are found
    count = result['count'] if result else 0
    return jsonify({'count': count})
