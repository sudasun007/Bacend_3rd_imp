from flask import Blueprint, request, jsonify, session, current_app
import mysql.connector

profile_bp = Blueprint('profile_bp', __name__)

@profile_bp.route('/profile', methods=['GET'])
def get_profile():
    if 'email' in session:
        email = session['email']
        try:
            # Establish MySQL connection directly
            conn = mysql.connector.connect(
                host='localhost',
                user='root',
                password='Sudheera30052807',
                database='Flaskapp'
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
        
    return jsonify({"error": "Unauthorized"}), 401

@profile_bp.route('/profile', methods=['PUT'])
def update_profile():
    
        data = request.get_json()
        print(data)
        f_name = data['F_name']
        l_name = data['L_name']
        email = data['Email']


        
        try:
            # Establish MySQL connection directly
            cursor = current_app.mysql.cursor(dictionary=True)
            cursor.execute("UPDATE Player SET F_name = %s, L_name = %s WHERE Email = %s", (f_name, l_name, email))

            current_app.mysql.commit()
            cursor.close()
            
            return jsonify({"message": "Profile updated successfully"})
        
        except Exception as e:
            return jsonify({"error": str(e)}), 500




@profile_bp.route('/profile/delete', methods=['POST'])
def delete_account():
    
        try:
            data=request.get_json()
            email=data['Email']
            # Establish MySQL connection directly
            cursor = current_app.mysql.cursor(dictionary=True)
            
            # Execute the DELETE query with the user's email
            cursor.execute("DELETE FROM Player WHERE Email = %s", (email,))
            current_app.mysql.commit()
            # Close cursor and connection
            cursor.close()
            
            # Return success message
            return jsonify({"message": "Account deleted successfully"})
        
        except Exception as e:
            # Handle any exceptions and return an error response
            return jsonify({"error": str(e)}), 500
       
