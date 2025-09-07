# Sử dụng Flask để parking api.
from flask import Flask, request, jsonify, Response
from model import get_chatbot_answer
import json
from db import save_chat  # nếu bạn muốn lưu lịch sử
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=["*"])

@app.route("/ask", methods=["POST"])
def get_news():
    data = request.get_json()
    question = data.get("question", "")

    # question = "giới thiệu về chương trình của SHB"
    answer = get_chatbot_answer(question)
    # save_chat(question, answer)  # nếu không muốn lưu có thể comment dòng này
    return Response(
        json.dumps({"response": answer}, ensure_ascii=False),
        content_type="application/json"
    )

if __name__ == "__main__":
    app.run(debug=True)
