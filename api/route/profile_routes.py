from flask import Blueprint, request, jsonify, session, current_app
import mysql.connector
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Define the blueprint for profile routes
profile_bp = Blueprint('profile_bp', __name__)

'''@profile_bp.route('/profile', methods=['GET'])
def get_profile():
    if 'email' in session:
        email = session['email']
        try:
            # Use environment variables for database connection details
            conn = mysql.connector.connect(
                host=os.getenv('MYSQL_DATABASE_HOST'),  # Get the host from .env
                user=os.getenv('MYSQL_DATABASE_USER'),  # Get the username from .env
                password=os.getenv('MYSQL_DATABASE_PASSWORD'),  # Get the password from .env
                database=os.getenv('MYSQL_DATABASE_DB')  # Get the database name from .env
            )
            cursor = conn.cursor(dictionary=True)
            
            # Query to fetch the player's latest rank and their name
            query = """
                SELECT Player.F_name, Player.L_name, Player.Email, 
                       COALESCE(Result.Rank_P, 0) AS Rank_P
                FROM Player
                LEFT JOIN Result ON Player.Email = Result.P_email
                WHERE Player.Email = %s
                ORDER BY Result.Date DESC
                LIMIT 1;
            """
            cursor.execute(query, (email,))
            user = cursor.fetchone()
            cursor.close()
            conn.close()

            if user:
                return jsonify({
                    "F_name": user['F_name'],
                    "L_name": user['L_name'],
                    "Email": user['Email'],
                    "Rank": user['Rank_P']
                })
            else:
                return jsonify({"error": "User not found"}), 404
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        
    return jsonify({"error": "Unauthorized"}), 401'''

@profile_bp.route('/profile', methods=['GET'])
def get_profile():
    if 'email' in session:
        email = session['email']
        try:
            # Use environment variables for database connection details
            conn = mysql.connector.connect(
                host=os.getenv('MYSQL_DATABASE_HOST'),
                user=os.getenv('MYSQL_DATABASE_USER'),
                password=os.getenv('MYSQL_DATABASE_PASSWORD'),
                database=os.getenv('MYSQL_DATABASE_DB')
            )
            cursor = conn.cursor(dictionary=True)
            
            # Query to fetch the player's latest rank and their name
            query = """
                SELECT Player.F_name, Player.L_name, Player.Email, 
                       COALESCE(Result.Rank_P, 0) AS Rank_P
                FROM Player
                LEFT JOIN Result ON Player.Email = Result.P_email
                WHERE Player.Email = %s
                ORDER BY Result.Date DESC
                LIMIT 1;
            """
            cursor.execute(query, (email,))
            user = cursor.fetchone()
            cursor.close()
            conn.close()

            if user:
                return jsonify({
                    "F_name": user['F_name'],
                    "L_name": user['L_name'],
                    "Email": user['Email'],
                    "Rank": user['Rank_P']
                })
            else:
                return jsonify({"error": "User not found"}), 404
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        
    return jsonify({"error": "Unauthorized"}), 401



@profile_bp.route('/profile', methods=['PUT'])
def update_profile():
    data = request.get_json()
    f_name = data['F_name']
    l_name = data['L_name']
    email = data['Email']
    
    try:
        # Establish MySQL connection using the same method
        conn = mysql.connector.connect(
            host=os.getenv('MYSQL_DATABASE_HOST'),
            user=os.getenv('MYSQL_DATABASE_USER'),
            password=os.getenv('MYSQL_DATABASE_PASSWORD'),
            database=os.getenv('MYSQL_DATABASE_DB')
        )
        cursor = conn.cursor(dictionary=True)
        cursor.execute("UPDATE Player SET F_name = %s, L_name = %s WHERE Email = %s", (f_name, l_name, email))

        conn.commit()
        cursor.close()
        conn.close()

        return jsonify({"message": "Profile updated successfully"})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@profile_bp.route('/profile/delete', methods=['POST'])
def delete_account():
    try:
        data = request.get_json()
        email = data['Email']

        # Establish MySQL connection using the same method
        conn = mysql.connector.connect(
            host=os.getenv('MYSQL_DATABASE_HOST'),
            user=os.getenv('MYSQL_DATABASE_USER'),
            password=os.getenv('MYSQL_DATABASE_PASSWORD'),
            database=os.getenv('MYSQL_DATABASE_DB')
        )
        cursor = conn.cursor(dictionary=True)

        # Execute the DELETE query with the user's email
        cursor.execute("DELETE FROM Player WHERE Email = %s", (email,))
        conn.commit()
        cursor.close()
        conn.close()

        return jsonify({"message": "Account deleted successfully"})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500



'''@profile_bp.route('/profile', methods=['GET'])
def get_profile():
    if 'email' in session:
        email = session['email']
        try:
            # Use environment variables for database connection details
            conn = mysql.connector.connect(
                host=os.getenv('MYSQL_DATABASE_HOST'),  # Get the host from .env
                user=os.getenv('MYSQL_DATABASE_USER'),  # Get the username from .env
                password=os.getenv('MYSQL_DATABASE_PASSWORD'),  # Get the password from .env
                database=os.getenv('MYSQL_DATABASE_DB')  # Get the database name from .env
            )
            cursor = conn.cursor(dictionary=True)
            
            query = """
                SELECT Player.F_name, Player.L_name, Player.Email, Result.Rank_P
                FROM Player
                LEFT JOIN Result ON Player.Email = Result.P_email
                WHERE Player.Email = %s
                ORDER BY Result.Date DESC
                LIMIT 1;
            """
            cursor.execute(query, (email,))
            user = cursor.fetchone()
            cursor.close()
            conn.close()

            if user:
                return jsonify({"F_name": user['F_name'], "L_name": user['L_name'], "Email": user['Email'], "Rank": user['Rank_P']})
            else:
                return jsonify({"error": "User not found"}), 404
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        
    return jsonify({"error": "Unauthorized"}), 401'''


@profile_bp.route('/get_user_rank', methods=['GET'])
def get_user_rank():
    try:
        player_email = request.args.get('player_email')

        if not player_email:
            return jsonify({'error': 'Player email is required'}), 400

        # Direct connection to the database
        conn = mysql.connector.connect(
            host=os.getenv('MYSQL_DATABASE_HOST'),
            user=os.getenv('MYSQL_DATABASE_USER'),
            password=os.getenv('MYSQL_DATABASE_PASSWORD'),
            database=os.getenv('MYSQL_DATABASE_DB')
        )
        cursor = conn.cursor()

        cursor.execute('''
            SELECT Rank_P
            FROM Result
            WHERE P_email = %s
            ORDER BY Date DESC
            LIMIT 1
        ''', (player_email,))

        result = cursor.fetchone()
        cursor.close()
        conn.close()

        if result:
            return jsonify({'rank': result[0]}), 200
        else:
            return jsonify({'error': 'Rank not found for this user'}), 404

    except Exception as e:
        logging.error(f"Error retrieving user rank: {e}")
        return jsonify({'error': 'Internal server error'}), 500



