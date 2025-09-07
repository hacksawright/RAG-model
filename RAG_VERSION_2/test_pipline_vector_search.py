from pymongo import MongoClient
import google.generativeai as genai

# ============================
# 1. Cấu hình API key Gemini
# ============================
genai.configure(api_key="AIzaSyA7EdJM8tB0w4EJM7yhybQtEi6-WibRfNI")

def get_embedding(text: str):
    """Sinh embedding từ Gemini."""
    result = genai.embed_content(
        model="models/embedding-001",
        content=text
    )
    return result['embedding']

# ============================
# 2. Kết nối Mongo Atlas
# ============================
client = MongoClient("mongodb+srv://nductrung779:12345@trr1.kfctgrg.mongodb.net/")
db = client["test"]          # thay bằng tên database
collection = db["test"]      # thay bằng tên collection

# ============================
# 3. Tạo embedding cho câu hỏi
# ============================
question = "CBT là gì?"
query_vector = get_embedding(question)

# ============================
# 4. Thực hiện Vector Search
# ============================
pipeline = [
    {
        "$vectorSearch": {
            "queryVector": query_vector,
            "path": "embedding",      # field chứa embedding trong document
            "numCandidates": 1000,
            "limit": 5,
            "index": "vector_index"   # tên index bạn đã tạo trong Atlas
        }
    },
    {
        "$project": {
            "text": 1,
            "_id": 0,
            "score": {"$meta": "vectorSearchScore"}
        }
    }
]

print("🔍 Đang chạy vector search...")
results = collection.aggregate(pipeline)

for r in results:
    print(r)

client.close()
