import requests
import pandas as pd
import json
from pymongo import MongoClient
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.embeddings import GPT4AllEmbeddings 
import google.generativeai as genai

n = ""
# API_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
# API_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L12-v2/pipeline/feature-extraction"
# API_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/sentence-similarity"

# model đọc được tiếng việt
API_URL1 = "https://router.huggingface.co/hf-inference/models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/pipeline/feature-extraction"

headers = {"Authorization": f"Bearer {n}"}

# API key Gemini
genai.configure(api_key="AIzaSyA7EdJM8tB0w4EJM7yhybQtEi6-WibRfNI")

def get_embedding(text: str):
    result = genai.embed_content(model="models/embedding-001", content=text)
    return result["embedding"]

def create_db_from_text():
    raw_text = """# Liệu pháp nhận thức hành vi (CBT) là gì?

*Posted on 18 Tháng Tám, 2022 (Cập nhật: 3 Tháng Ba, 2025) by Viện Tâm lý Giáo dục VCP*

Liệu pháp nhận thức hành vi hay còn được gọi tắt là **CBT – Cognitive Behavioral Therapy** là một trong các biện pháp can thiệp xã hội tập trung chủ yếu vào việc thách thức và tiếp nhận các biến dạng nhận thức không có ích (như thái độ, niềm tin, suy nghĩ và hành vi), giúp cải thiện điều tiết cảm xúc và phát triển những chiến lược để đối phó cá nhân nhằm khắc phục và giải quyết những vấn đề của hiện tại.

Lúc đầu liệu pháp này được nghiên cứu và thiết kế nhằm cải thiện các triệu chứng của trầm cảm nhưng về sau được mở rộng và sử dụng phổ biến hơn trong quá trình cải thiện một số vấn đề về sức khỏe tâm thần, trong đó có lo âu.

Nói một cách dễ hiểu hơn thì CBT chính là một liệu pháp được sử dụng để tìm kiếm và cải thiện các dạng suy nghĩ, hành vi tiêu cực trong nhận thức của mỗi người về một sự kiện hay tình huống nào đó tạo ra các vấn đề về tâm lý, những mối quan hệ hoặc rối nhiễu về khía cạnh tinh thần của cá nhân đó.

Khi một sự kiện, tình huống nào đó bắt đầu xảy ra sẽ kích thích suy nghĩ của một cá nhân. Suy nghĩ này sẽ tác động lên cảm xúc dẫn đến việc cá nhân đó sẽ thực hiện một hành động ra bên ngoài. Và hành động, suy nghĩ, cảm xúc đó sẽ ảnh hưởng lên thể lý của cá nhân đó.

Đôi lúc chúng ta cũng sẽ chịu sự tác động của một số bệnh thực thể đối với suy nghĩ và cảm xúc, hành động của chính mình. Do đó, có thể nhận thấy hành vi, cảm xúc, suy nghĩ, thể lý có sự tác động và tương tác qua lại lẫn nhau.

Liệu pháp nhận thức hành vi thường sẽ được áp dụng trong thời gian ngắn và hỗ trợ người bệnh đối mặt với những vấn đề cụ thể đang xảy ra trong hiện tại. Trong suốt thời gian chữa trị, người bệnh sẽ được hướng dẫn cách xác định và thay đổi các suy nghĩ sai lệch làm ảnh hưởng xấu đến cảm xúc và hành vi của chính mình.

Có 3 điểm cơ bản để xây dựng nên lý thuyết về nhận thức:

*   Hiểu và biết rõ về quan điểm mà bản thân lựa chọn tác động đến tâm trạng của chính mình.
*   Tâm trạng và cách thức suy nghĩ có mối liên hệ mật thiết với nhau vì thế nếu suy nghĩ được thay đổi thì tâm trạng cũng sẽ biến đổi theo và ngược lại.
*   Học cách làm việc dựa vào niềm tin và suy nghĩ của bản thân.

Các chuyên gia cho biết rằng, mô hình của CBT sẽ dựa trên sự kết hợp của những nguyên tắc cơ bản có từ tâm lý học hành vi, nhận thức. Liệu pháp này hoàn toàn khác với cách tiếp cận lịch sử của tâm lý trị liệu.

Ví dụ như khi sử dụng liệu pháp phân tâm học các nhà trị liệu sẽ khai thác và tìm kiếm ý nghĩa vô thức phía sau những hành vi và bắt đầu hình thành một chẩn đoán. Còn đối với CBT thì sẽ tập trung vào vấn đề và giúp người bệnh định hướng hành động, tức là nó sẽ áp dụng để giải quyết các vấn đề cụ thể có liên quan đến một số rối loạn tâm thần đã được chẩn đoán.

Các nhà trị liệu sẽ đóng vai trò là một người hỗ trợ khách hàng để giúp họ tìm kiếm và thực hiện những chiến lược tốt nhằm giải quyết những mục tiêu đã được xác định, bên cạnh đó giúp thuyên giảm các triệu chứng mà rối loạn tâm thần gây ra.

Liệu pháp nhận thức hành vi sẽ dựa trên niềm tin rằng những biến dạng hành vi và nhận thức không tích cực đóng một vai trò trong quá trình duy trì và phát triển các rối loạn tâm lý, đồng thời các triệu chứng có liên quan cũng sẽ được khắc phục bằng biện pháp dạy những kỹ năng xử lý thông tin cùng cơ chế đối phó."""

    # Chia nho van ban
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)

    chunks = text_splitter.split_text(raw_text)
    # for i, chunk in enumerate(chunks):
    #     print(f"Chunk {i}:\n{chunk}\n")
    # for i in range(1, len(chunks)):
    #     prev_chunk = chunks[i-1]
    #     curr_chunk = chunks[i]
    #     print(f"Overlap {i-1}->{i}: {repr(prev_chunk[-30:])} | {repr(curr_chunk[:30])}")
    # embedding_model = GPT4AllEmbeddings(model_file = "all-MiniLM-L6-v2-bf16.gguf")
    # Embeding va luu vao mongodb
    for text in chunks:
        # embedding = embedding_model.embed_query(text)
        embedding = get_embedding(text)
        if embedding:
            collection.insert_one({
                "text": text,
                "embedding": embedding
            })

client=MongoClient("mongodb+srv://nductrung779:12345@trr1.kfctgrg.mongodb.net/")

db = client["test"]
collection = db["test"]
#Xóa dữ liệu cũ trước khi insert mới
collection.drop() 

create_db_from_text()

print("Đã lưu embeddings từ API Hugging Face vào MongoDB.")