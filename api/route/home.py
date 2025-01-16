from distutils import debug
from http import HTTPStatus
from logging import info
from xml.dom import ValidationErr
from flask import Blueprint, request, jsonify, current_app, session
from flasgger import swag_from
from werkzeug.security import generate_password_hash, check_password_hash
from api.schema.player import PlayerSchema, LoginSchema



home_api = Blueprint('api', __name__)


@home_api.route('/')
@swag_from({
    'responses': {
        HTTPStatus.OK.value: {
            'description': 'Welcome to the Flask Starter Kit',
            'schema': {}
        }
    }
})
def welcome():
    return jsonify({'message': 'Welcome to the Flask Starter Kit'}), HTTPStatus.OK


@home_api.route('/signup', methods=['POST'])
@swag_from({
    'parameters': [
        {
            'name': 'body',
            'in': 'body',
            'required': True,
            'schema': PlayerSchema
        }
    ],
    'responses': {
        HTTPStatus.CREATED.value: {
            'description': 'Player successfully registered',
            'schema': PlayerSchema
        },
        HTTPStatus.BAD_REQUEST.value: {
            'description': 'Invalid input'
        }
    }
})
def signup():
    try:
        data = request.get_json()
        schema = PlayerSchema()

        if 'Active_status' in data:
            del data['Active_status']

        validated_data = schema.load(data)

        # Check if email already exists
        cursor = current_app.mysql.cursor(dictionary=True)
        cursor.execute("SELECT * FROM Player WHERE Email = %s",
                       (validated_data['Email'],))
        existing_player = cursor.fetchone()

        if existing_player:
            return jsonify({'message': 'Email already registered'}), HTTPStatus.BAD_REQUEST

        # Hash the password
        hashed_password = generate_password_hash(validated_data['Password'])

        # Insert new player
        cursor.execute(
            "INSERT INTO Player (Email, F_name, L_name, Password) VALUES (%s, %s, %s, %s)",
            (validated_data['Email'], validated_data['F_name'],
             validated_data['L_name'], hashed_password)
        )
        current_app.mysql.commit()
        cursor.close()

        new_player = {
            'Email': validated_data['Email'],
            'F_name': validated_data['F_name'],
            'L_name': validated_data['L_name'],
            'Password': hashed_password
        }

        return jsonify(new_player), HTTPStatus.CREATED
    except ValidationErr as err:
        return jsonify(err), HTTPStatus.BAD_REQUEST


@home_api.route('/login', methods=['POST'])
@swag_from({
    'parameters': [
        {
            'name': 'body',
            'in': 'body',
            'required': True,
            'schema': LoginSchema
        }
    ],
    'responses': {
        HTTPStatus.OK.value: {
            'description': 'Player successfully logged in',
            'schema': PlayerSchema
        },
        HTTPStatus.UNAUTHORIZED.value: {
            'description': 'Invalid email or password'
        }
    }
})
def login():
    try:
        data = request.get_json()
        schema = LoginSchema()

        if 'Active_status' in data:
            del data['Active_status']

        validated_data = schema.load(data)
        
        print(validated_data)

        # Check if the player exists
        cursor = current_app.mysql.cursor(dictionary=True)
        cursor.execute("SELECT * FROM Player WHERE Email = %s",
                       (validated_data['Email'],))

        player = cursor.fetchone()

        if player and check_password_hash(player['Password'], validated_data['Password']):
            # Set session data upon successful login
            session['email'] = player['Email']  # Store email in session
            session.permanent = True  # Make the session persist beyond browser closure if desired

            # Update Active_status to True (1)
            cursor = current_app.mysql.cursor(dictionary=True)
            cursor.execute(
                "UPDATE Player SET Active_status = TRUE WHERE Email = %s", (validated_data['Email'],))
            current_app.mysql.commit()
            cursor.close()
            
            if player['Active_status'] == 1:
                player['Active_status'] = 'true' 
            else:
                player['Active_status'] = 'false'

            return jsonify({'message': 'Login successful', 'player': player}), HTTPStatus.OK
        else:
            return jsonify({'message': 'Invalid email or password'}), HTTPStatus.UNAUTHORIZED
    except ValidationErr as err:
        return jsonify(err), HTTPStatus.BAD_REQUEST


@home_api.route('/logout', methods=['POST'])
@swag_from({
    'responses': {
        HTTPStatus.OK.value: {
            'description': 'Player successfully logged out'
        },
        HTTPStatus.UNAUTHORIZED.value: {
            'description': 'No active session found'
        }
    }
})
def logout():
    data = request.get_json()
    email = data.get('Email')

    if not email:
        return jsonify({'message': 'Email is required'}), HTTPStatus.BAD_REQUEST

    # Check if the player exists
    cursor = current_app.mysql.cursor(dictionary=True)
    cursor.execute("SELECT * FROM Player WHERE Email = %s", (email,))
    player = cursor.fetchone()

    if not player:
        cursor.close()
        return jsonify({'message': 'No active session found'}), HTTPStatus.UNAUTHORIZED

    # Update Active_status to False (0)
    cursor.execute(
        "UPDATE Player SET Active_status = FALSE WHERE Email = %s", (email,))
    current_app.mysql.commit()

    # Close cursor
    cursor.close()

    # Clear session data
    session.pop('email', None)

    return jsonify({'message': 'Logout successful'}), HTTPStatus.OK

##newly added validate email

@home_api.route('/validate_email', methods=['POST'])
@swag_from({
    'parameters': [
        {
            'name': 'body',
            'in': 'body',
            'required': True,
            'schema': {
                'type': 'object',
                'properties': {
                    'email': {
                        'type': 'string'
                    }
                },
                'required': ['email']
            }
        }
    ],
    'responses': {
        HTTPStatus.OK.value: {
            'description': 'Email is registered',
            'schema': {
                'type': 'object',
                'properties': {
                    'message': {
                        'type': 'string'
                    }
                }
            }
        },
        HTTPStatus.NOT_FOUND.value: {
            'description': 'Email not registered'
        }
    }
})
def validate_email():
    data = request.get_json()
    email = data.get('email')

    cursor = current_app.mysql.cursor(dictionary=True)
    cursor.execute("SELECT * FROM Player WHERE Email = %s", (email,))
    player = cursor.fetchone()
    cursor.close()

    if player:
        session['reset_email'] = email  # Store email in session this is new addded
        return jsonify({'message': 'Email is registered'}), HTTPStatus.OK
    else:
        return jsonify({'message': 'Email not registered'}), HTTPStatus.NOT_FOUND
    
    

'''@home_api.route('/reset-password', methods=['POST'])
@swag_from({
    'parameters': [
        {
            'name': 'body',
            'in': 'body',
            'required': True,
            'schema': {
                'type': 'object',
                'properties': {
                    'email': {'type': 'string'},
                    'new_password': {'type': 'string'},
                    'confirm_password': {'type': 'string'}
                },
                'required': ['email', 'new_password', 'confirm_password']
            }
        }
    ],
    'responses': {
        HTTPStatus.OK.value: {
            'description': 'Password reset successful',
            'schema': {
                'type': 'object',
                'properties': {
                    'message': {'type': 'string'}
                }
            }
        },
        HTTPStatus.BAD_REQUEST.value: {
            'description': 'Invalid input'
        }
    }
})
def resetPassword():
    try:
        data = request.get_json()
        
        email = data.get('email')
        new_password = data.get('new_password')
        confirm_password = data.get('confirm_password')
        
        if not email or not new_password or not confirm_password:
            return jsonify({'message': 'Email, new password, and confirm password are required'}), HTTPStatus.BAD_REQUEST
        
        if new_password != confirm_password:
            return jsonify({'message': 'Passwords do not match'}), HTTPStatus.BAD_REQUEST
        
        cursor = current_app.mysql.cursor(dictionary=True)
        cursor.execute("SELECT * FROM Player WHERE Email = %s", (email,))
        player = cursor.fetchone()
        
        if not player:
            cursor.close()
            return jsonify({'message': 'Email not registered'}), HTTPStatus.NOT_FOUND
        
        hashed_password = generate_password_hash(new_password)
        
        cursor.execute("UPDATE Player SET Password = %s WHERE Email = %s", (hashed_password, email))
        current_app.mysql.commit()
        cursor.close()
        
        return jsonify({'message': 'Password reset successful'}), HTTPStatus.OK
    
    except ValidationErr as err:
        return jsonify(err), HTTPStatus.BAD_REQUEST
    except Exception as e:
        return jsonify({'message': str(e)}), HTTPStatus.INTERNAL_SERVER_ERROR  '''

'''@home_api.route('/reset-password', methods=['POST'])
def reset_password():
    try:
        data = request.get_json()
        
        email = data.get('email')
        new_password = data.get('new_password')
        confirm_password = data.get('confirm_password')
        
        if not email or not new_password or not confirm_password:
            return jsonify({'message': 'Email, new password, and confirm password are required'}), HTTPStatus.BAD_REQUEST
        
        if new_password != confirm_password:
            return jsonify({'message': 'Passwords do not match'}), HTTPStatus.BAD_REQUEST
        
        # Connect to your database
        cursor = current_app.mysql.cursor(dictionary=True)
        
        # Check if the email exists
        cursor.execute("SELECT * FROM Player WHERE Email = %s", (email,))
        player = cursor.fetchone()
        
        if not player:
            cursor.close()
            return jsonify({'message': 'Email not registered'}), HTTPStatus.NOT_FOUND
        
        # Hash the new password
        hashed_password = generate_password_hash(new_password)
        
        # Update the password in the database
        cursor.execute("UPDATE Player SET Password = %s WHERE Email = %s", (hashed_password, email))
        current_app.mysql.commit()
        cursor.close()
        
        return jsonify({'message': 'Password reset successful'}), HTTPStatus.OK
    
    except Exception as e:
        return jsonify({'message': str(e)}), HTTPStatus.INTERNAL_SERVER_ERROR'''




'''@home_api.route('/reset_password', methods=['POST'])
@swag_from({
    'parameters': [
        {
            'name': 'body',
            'in': 'body',
            'required': True,
            'schema': {
                'type': 'object',
                'properties': {
                    'new_password': {'type': 'string'},
                    'confirm_password': {'type': 'string'}
                },
                'required': ['new_password', 'confirm_password']
            }
        }
    ],
    'responses': {
        HTTPStatus.OK.value: {
            'description': 'Password reset successful',
            'schema': {
                'type': 'object',
                'properties': {
                    'message': {'type': 'string'}
                }
            }
        },
        HTTPStatus.BAD_REQUEST.value: {
            'description': 'Invalid input'
        }
    }
})
def reset_password():
    try:
        data = request.get_json()
        new_password = data.get('new_password')
        confirm_password = data.get('confirm_password')
        
        email = session.get('reset_email')  # Get the email from the session

        if not email:
            return jsonify({'message': 'Email not found in session'}), HTTPStatus.BAD_REQUEST

        if not new_password or not confirm_password:
            return jsonify({'message': 'New password and confirm password are required'}), HTTPStatus.BAD_REQUEST
        
        if new_password != confirm_password:
            return jsonify({'message': 'Passwords do not match'}), HTTPStatus.BAD_REQUEST
        
        # Hash the new password and update it in the database
        hashed_password = generate_password_hash(new_password)
        cursor = current_app.mysql.cursor(dictionary=True)
        cursor.execute("UPDATE Player SET Password = %s WHERE Email = %s", (hashed_password, email))
        current_app.mysql.commit()
        cursor.close()
        
        # Clear the session email
        session.pop('reset_email', None)
        
        return jsonify({'message': 'Password reset successful'}), HTTPStatus.OK
    
    except Exception as e:
        return jsonify({'message': str(e)}), HTTPStatus.INTERNAL_SERVER_ERROR'''

@home_api.route('/password_reset', methods=['POST'])
@swag_from({
    'parameters': [
        {
            'name': 'body',
            'in': 'body',
            'required': True,
            'schema': {
                'type': 'object',
                'properties': {
                    'new_password': {'type': 'string'},
                    'confirm_password': {'type': 'string'}
                },
                'required': ['new_password', 'confirm_password']
            }
        }
    ],
    'responses': {
        HTTPStatus.OK.value: {
            'description': 'Password reset successful',
            'schema': {
                'type': 'object',
                'properties': {
                    'message': {'type': 'string'}
                }
            }
        },
        HTTPStatus.BAD_REQUEST.value: {
            'description': 'Invalid input or password mismatch',
            'schema': {
                'type': 'object',
                'properties': {
                    'message': {'type': 'string'}
                }
            }
        },
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {
            'description': 'Internal server error',
            'schema': {
                'type': 'object',
                'properties': {
                    'message': {'type': 'string'}
                }
            }
        }
    }
})
def password_reset():
    try:
        data = request.get_json()
        new_password = data.get('new_password')
        confirm_password = data.get('confirm_password')
        
        email = data.get('email')  # Get the email from the session

        if not email:
            return jsonify({'message': 'Email not found in session'}), HTTPStatus.BAD_REQUEST
        
        if not new_password or not confirm_password:
            return jsonify({'message': 'New password and confirm password are required'}), HTTPStatus.BAD_REQUEST
        
        if new_password != confirm_password:
            return jsonify({'message': 'Passwords do not match'}), HTTPStatus.BAD_REQUEST
        
        # Hash the new password and update it in the database
        hashed_password = generate_password_hash(new_password)
        cursor = current_app.mysql.cursor(dictionary=True)
        cursor.execute("UPDATE Player SET Password = %s WHERE Email = %s", (hashed_password, email))
        current_app.mysql.commit()
        cursor.close()
        
        # Clear the session email
        session.pop('reset_email', None)
        
        return jsonify({'message': 'Password reset successful'}), HTTPStatus.OK
    
    except Exception as e:
        current_app.logger.error(f'Error resetting password: {str(e)}')  # Log the error for debugging
        return jsonify({'message': 'Internal server error'}), HTTPStatus.INTERNAL_SERVER_ERROR


