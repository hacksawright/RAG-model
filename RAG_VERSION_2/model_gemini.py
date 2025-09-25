# model.py
from pymongo import MongoClient
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
import google.generativeai as genai

# Cấu hình Gemini API key
genai.configure(api_key="")

# 2. Hàm get embedding
def get_embedding(text: str):
    result = genai.embed_content(
        model="models/embedding-001",
        content=text
    )
    # API trả về mảng trong result["embedding"]["values"]
    return result['embedding']

# 3. Wrapper cho VectorStore
class GoogleEmbeddingWrapper:
    def embed_documents(self, texts):
        return [get_embedding(t) for t in texts]

    def embed_query(self, text):
        return get_embedding(text)

embedding_model = GoogleEmbeddingWrapper()

# 4. Khởi tạo VectorStore (MongoDB Atlas)
vector_db = MongoDBAtlasVectorSearch.from_connection_string(
    connection_string="mongodb+srv://nductrung779:K3p4ME0oIMoc3ipa@cluster0.mdfkkus.mongodb.net/",
    namespace="AI_PARTNER.embedding",
    embedding=embedding_model,
    text_key="text",  # nếu bạn lưu thêm text
    index_name="vector_index"
)
retriever = vector_db.as_retriever(search_kwargs={"k": 10}, max_tokens_limit=1024)

# Tạo prompt
prompt_template = PromptTemplate(
    template="""Sử dụng thông tin sau đây để trả lời câu hỏi:\n
    {context}. \nCâu hỏi: {question}""",
    input_variables=["context", "question"]
)

# # Tạo prompt
# prompt_template = PromptTemplate(
#     template="""
#     Bạn là một trợ lý chỉ được phép sử dụng thông tin sau đây để trả lời:
#     -----
#     {context}
#     -----
#     Câu hỏi: {question}

#     Nếu không có thông tin phù hợp trong dữ liệu, hãy trả lời:
#     "❌ Không có thông tin trong dữ liệu."
#     """,
#     input_variables=["context", "question"]
# )

# Thêm một dictionary để lưu trữ các từ đồng nghĩa
synonyms_mapping = {
    "CBT": ["liệu pháp nhận thức hành vi", "Cognitive Behavioral Therapy"]
    # Thêm các từ đồng nghĩa khác tại đây
}

# Hàm xử lý truy vấn
def get_chatbot_answer(question: str) -> str:
    # 1. Tìm và thay thế các từ đồng nghĩa trong câu hỏi
    expanded_question = question
    for key, values in synonyms_mapping.items():
        # Tìm và thay thế từ đồng nghĩa bằng từ gốc
        for synonym in values:
            if synonym in question.lower():
                expanded_question = expanded_question.replace(synonym, key)
                break
    docs = retriever.invoke(question)
    if not docs:
        return "❌ Không tìm thấy thông tin trong dữ liệu."
    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = prompt_template.invoke({"context": context, "question": question}).to_string()

    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    return response.text

# question = "CBT là gì"
# docs = retriever.invoke(question)
# if docs:
#     context = "\n\n".join(doc.page_content for doc in docs)
#     print("Context:", context)
# else:
#     print("Không tìm thấy tài liệu.")
