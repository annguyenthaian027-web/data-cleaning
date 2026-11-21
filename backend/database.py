import os

from pymongo import MongoClient

MONGO_URL = os.getenv("MONGO_URL", "mongodb://127.0.0.1:27017")

# USE_MOCK_DB=1: dùng MongoDB giả lập trong bộ nhớ (chạy thử không cần cài MongoDB, mất dữ liệu khi tắt)
if os.getenv("USE_MOCK_DB") == "1":
    import mongomock
    client = mongomock.MongoClient()
else:
    client = MongoClient(MONGO_URL, serverSelectionTimeoutMS=5000)

db = client["data_cleaning_db"]
users_collection = db["users"]


def check_database_connection():
    try:
        client.admin.command("ping")
        return True
    except Exception as e:
        print("Loi ket noi MongoDB:", e)
        return False
