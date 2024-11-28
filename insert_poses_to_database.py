import mysql.connector

# List of poses with details and public URLs
poses = [
    ('zenkutsuDachi_awaseTsuki(leftLeg)', 'Standing stance with feet shoulder-width apart, left leg forward, executing a combined punch.', 'https://drive.google.com/uc?id=1uuRfbeOLePQT-_f-W5b_Frb2VbjYbgVT'),
    ('shikoDachi_gedanBarai(front)', 'Sumo stance with feet wide apart, performing a downward block to the front.', 'https://drive.google.com/uc?id=1Pc1dRjI0tmrVsojwtHttzNh-ZkHhFNmo'),
    ('zenkutsuDachi_empiUke(right)', 'Front stance with right arm executing an elbow block.', 'https://drive.google.com/uc?id=12_zvdQ99Q4yY95kYi_ylfzpjob4j9MXS'),
    ('zenkitsuDahi_empiUke(left)', 'Front stance with left arm executing an elbow block.', 'https://drive.google.com/uc?id=1Zer7r3k2IYfh_YnuuIc2SncGkO4EfrZ2'),
    ('zenkutsuDachi_chudanTsuki', 'Front stance with a middle punch.', 'https://drive.google.com/uc?id=1hUOWIxnZIdomIn-mPkRIlA_xbNqBNwRi'),
    ('shikoDach_gedaiBarai(left)', 'Details for shikoDach_gedaiBarai(left)', 'https://drive.google.com/uc?id=1WKWSyfl1nejtU4GOM5f1BWlone0esuPy'),
    ('shikoDachi_GedanBarai(right)', 'Details for shikoDachi_GedanBarai(right)', 'https://drive.google.com/uc?id=1WgUTJ-h7pyx9F5KHkn__j4Be7rRYSKlb'),
    ('zenkutsuDachi_awaseTsuki(rightLeg)', 'Details for zenkutsuDachi_awaseTsuki(rightLeg)', 'https://drive.google.com/uc?id=1cfmlyo62xxtE-lQe9o8F2UrfHnn4P-SH'),
    ('sotoUke_maeGeri(right)', 'Details for sotoUke_maeGeri(right)', 'https://drive.google.com/uc?id=1tX6VcsC9fkmksSJU0AfcK_1XF2geFgru'),
    ('sotoUke_maeGeri(left)', 'Details for sotoUke_maeGeri(left)', 'https://drive.google.com/uc?id=1QUZGU_1_A4Ppau5uiiNS0IVH5_4XO9hN'),
    ('motoDachi_sotoUke(left2)', 'Details for motoDachi_sotoUke(left2)', 'https://drive.google.com/uc?id=1gasQHrx1ZA_A2NHOzIKaiUg6mdcHo_ji'),
    ('motoDachi_sotoUke(right)', 'Details for motoDachi_sotoUke(right)', 'https://drive.google.com/uc?id=1KylrNBPGBZdbOl7oFaVqxTiff75dDZGD'),
    ('motoDachi_sotoUke(left)', 'Details for motoDachi_sotoUke(left)', 'https://drive.google.com/uc?id=1Evujtzh2SDvah5WnBq7rOE0UcE_ca8G6'),
    ('motoDachi_jodanTsuki(left)', 'Details for motoDachi_jodanTsuki(left)', 'https://drive.google.com/uc?id=16i2AVXX1gobHMDB6OU51437jHLv1xcMy'),
    ('motoDachi_jodanTsuki(right)', 'Details for motoDachi_jodanTsuki(right)', 'https://drive.google.com/uc?id=1qbXW_8ZehqndfJoEGppnsf77ePm1vhU2'),
    ('motoDachi_ageUke(right)', 'Details for motoDachi_ageUke(right)', 'https://drive.google.com/uc?id=1XUizHboNbdfWz3BX-ME5IREMeNm-rQh3'),
    ('motoDachi_ageUke(left)', 'Details for motoDachi_ageUke(left)', 'https://drive.google.com/uc?id=1OHT9wfGB9r81vHscDpVnHZa_v5CHu1kk'),
    ('hachijiDachi_jodanYoko(left)', 'Details for hachijiDachi_jodanYoko(left)', 'https://drive.google.com/uc?id=10ZFMiTJZ5z7ROvVGohDqB0cGpyAWxu20'),
    ('hachijiDachi_jidanYoko(right)', 'Details for hachijiDachi_jidanYoko(right)', 'https://drive.google.com/uc?id=1knHwEGtLSp-papsPmJ7C8W_sIw09R1Tu'),
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
    insert_query = "INSERT INTO Correct_pose (Correct_Pose_name, Pose_Details, Img_link) VALUES (%s, %s, %s)"
    cursor.executemany(insert_query, poses)

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
