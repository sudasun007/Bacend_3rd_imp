from flask import Blueprint, request, jsonify
import mysql.connector
import os
from dotenv import load_dotenv

# Create a new blueprint for the count results API
count_results_api = Blueprint('count_results_api', __name__)

# Initialize MySQL connection (use your app's configuration or directly connect)
def get_mysql_connection():
    connection = mysql.connector.connect(
        host=os.getenv('MYSQL_DATABASE_HOST'),
        user=os.getenv('MYSQL_DATABASE_USER'),
        password=os.getenv('MYSQL_DATABASE_PASSWORD'),
        database=os.getenv('MYSQL_DATABASE_DB')
    )
    return connection

@count_results_api.route('/count_results', methods=['POST'])
def count_results():
    # Get P_email from the request body
    data = request.get_json()
    p_email = data.get('P_email')

    if not p_email:
        return jsonify({'error': 'P_email is required'}), 400

    # Get the MySQL connection and create a cursor
    connection = get_mysql_connection()
    cursor = connection.cursor(dictionary=True)
    
    # Query the database to count records matching the P_email
    query = """
        SELECT Result_id 
        FROM Result 
        WHERE P_email = %s 
        ORDER BY Date DESC 
        LIMIT 1
    """
    cursor.execute(query, (p_email,))
    result = cursor.fetchone()

    # Close the cursor and connection
    cursor.close()
    connection.close()

    # Return the count as a JSON response
    if result:
        return jsonify({'Result_id': result['Result_id']})
    else:
        return jsonify({'message': 'No records found for the provided P_email'}), 404

