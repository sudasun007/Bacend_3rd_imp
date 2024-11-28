import mysql.connector

# List of guide videos with details and public URLs
guide_videos = [
    ('Basics of Goju Ryu Seiwakai Gekisai Dai Ichi', 'https://drive.google.com/uc?id=1KxaFYg_iZC4x5LzTJGFi6klHBFrXCttI'),
    ('Gekisai Dai Ichi', 'https://drive.google.com/uc?id=1cLuiLryVOvSJj7dw4BvW4QmwfAFxPPI3'),
    ('Gekisai Dai Ichi Kata Goju Ryu Karate by Sensei Davy Wijaya', 'https://drive.google.com/uc?id=1ndnS7Vxm3JSKWlCMH-mk4CK03nS0y6r-'),
    ('Goju Ryu Kata - Gekisai- Dai- Ichi 撃砕大一', 'https://drive.google.com/uc?id=1JBonKEovzQ-xi8F1bWypMg6xakCOvKEO'),
    ('Goju-ryu Kata Gekisai Dai Ichi (Slow)', 'https://drive.google.com/uc?id=1onAYy2V_4l9W_baI-He__C6yAG34gPpQ')
]

# Database configuration
config = {
    'user': 'root',
    'password': 'Sudheera30052807',
    'host': 'localhost',
    'database': 'Flaskapp'
}

# Connect to MySQL
try:
    connection = mysql.connector.connect(**config)
    cursor = connection.cursor()

    # Execute INSERT INTO statements
    insert_query = "INSERT INTO Guide_video (Guide_video_name, Video_link) VALUES (%s, %s)"
    cursor.executemany(insert_query, guide_videos)

    # Commit changes
    connection.commit()
    print("Records inserted successfully")

except mysql.connector.Error as error:
    print(f"Error inserting records: {error}")

finally:
    if connection.is_connected():
        cursor.close()
        connection.close()
        print("MySQL connection is closed")
