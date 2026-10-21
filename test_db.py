
from db import get_db_connection

connection = get_db_connection()
cursor = connection.cursor()

cursor.execute(
    """
    INSERT INTO chat_history (question, answer, source_document)
    VALUES (%s, %s, %s)
    """,
    (
        "What was the total project budget?",
        "The total project budget was ₹24 lakh.",
        "RAG_Test_Document.pdf",
    ),
)

connection.commit()
print("Test chat saved successfully!")

cursor.close()
connection.close()
