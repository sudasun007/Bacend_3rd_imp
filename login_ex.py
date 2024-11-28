import requests

# Login endpoint details
login_url = 'http://127.0.0.1:5000/api/login'  # Corrected the URL by removing the extra slash
headers = {'Content-Type': 'application/json'}
login_data = {'Email': 'dasun@example.com', 'Password': '123'}

# Perform login request
login_response = requests.post(login_url, json=login_data, headers=headers)

# Check if login was successful
if login_response.status_code == 200:
    print("Login successful")

    # Extract session cookie from response
    session_cookie = login_response.cookies.get('session')

    if session_cookie:
        # Example: Use the session cookie in a GET request to profile endpoint
        profile_url = 'http://127.0.0.1:5000/api/profile'
        profile_headers = {'Cookie': f'session={session_cookie}'}

        profile_response = requests.get(profile_url, headers=profile_headers)

        if profile_response.status_code == 200:
            print("Profile request successful")
            print(profile_response.json())  # Print profile data received
        else:
            print(f"Profile request failed: {profile_response.json()}")
    else:
        print("Login successful but session cookie not found")
else:
    try:
        print(f"Login failed: {login_response.json()}")
    except ValueError:
        print("Login failed: Unable to decode JSON response")
