# db.py
# Lưu câu hỏi, câu trả lời và thời gian khi chat vào mongodb
from pymongo import MongoClient
from datetime import datetime

client = MongoClient("mongodb+srv://nductrung779:12345@trr1.kfctgrg.mongodb.net/")
db = client["chatbot"]
chat_collection = db["history"]

def save_chat(question, answer):
    chat_collection.insert_one({
        "question": question,
        "answer": answer,
        "timestamp": datetime.now()
    })
