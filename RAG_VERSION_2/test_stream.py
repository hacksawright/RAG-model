
import time
from flask import Flask, request, jsonify, Response
from model import get_chatbot_answer
import json
from db import save_chat  # nếu bạn muốn lưu lịch sử
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Fake bot trả lời dần từng dòng
def generate_response_stream(question):
    fake_lines = [
        f"Bạn vừa hỏi: {question}",
        "Để mình suy nghĩ một chút...",
        "Câu trả lời là...",
        "Một cộng một bằng hai.",
        "Cảm ơn bạn đã hỏi!"
    ]
    for line in fake_lines:
        yield f"data: {line}\n\n"
        time.sleep(0.6)  # giả lập gõ từ từ


# @app.route("/ask-stream", methods=["GET"])
# def ask_stream():
#     data = request.get_json()
#     question = data.get("question", "")
#     answer = get_chatbot_answer(question)

#     def stream_answer():
#         # Tách câu theo dấu chấm. Có thể dùng split("\n") nếu muốn chia dòng.
#         for part in answer.split('. '):  
#             yield f"data: {part.strip()}.\n\n"
#             time.sleep(0.4)  # giả lập hiệu ứng typing

#     return Response(stream_answer(), mimetype='text/event-stream')

@app.route("/ask-stream", methods=["GET"])
def ask_stream():
    # Lấy câu hỏi từ URL: ?question=...
    # question = request.args.get("question", "")
    
    # if not question:
    #     return "Thiếu câu hỏi", 400

    question = "1+1 bằng mấy"
    answer = get_chatbot_answer(question)

    def stream_answer():
        for part in answer.split('. '):
            yield f"data: {part.strip()}.\n\n"
            time.sleep(0.4)

    return jsonify({"response": answer}) 


if __name__ == "__main__":
    app.run(debug=True)
