
import os
import mysql.connector
from dotenv import load_dotenv

load_dotenv()

def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user=os.getenv("MYSQL_USER", "root"),
        password=os.getenv("MYSQL_PASSWORD"),
        database="intelligence_hub"
    )

if __name__ == "__main__":
    print("Testing MySQL connection...")

    try:
        connection = get_db_connection()
        print("MySQL connection successful!")
        connection.close()
    except mysql.connector.Error as error:
        print("MySQL connection failed:", error)