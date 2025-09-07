# model.py

from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_community.embeddings import GPT4AllEmbeddings
import google.generativeai as genai

# Cấu hình Gemini API key
genai.configure(api_key="AIzaSyABRVpyQlZzD3oxDY6UrXN-r6Bfhe8R83w")

# Load vector DB
embedding_model = GPT4AllEmbeddings(model_file="models/all-MiniLM-L6-v2-f16.gguf")
vector_db = MongoDBAtlasVectorSearch.from_connection_string(
    connection_string="mongodb+srv://nductrung779:12345@trr1.kfctgrg.mongodb.net/7017",
    namespace="chatbot_embeddings.sentences",
    embedding=embedding_model,
    # text_key="content"
)
retriever = vector_db.as_retriever(search_kwargs={"k": 3}, max_tokens_limit=1024)

# Tạo prompt
prompt_template = PromptTemplate(
    template="""Sử dụng thông tin sau đây để trả lời câu hỏi:\n
    {context}. \nCâu hỏi: {question}""",
    input_variables=["context", "question"]
)

# Hàm xử lý truy vấn
def get_chatbot_answer(question: str) -> str:
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = prompt_template.invoke({"context": context, "question": question}).to_string()

    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    return response.text

# question = "Đại số có bao nhiêu tín?"
# docs = retriever.invoke(question)
# context = "\n\n".join(doc.page_content for doc in docs)
# prompt = prompt_template.invoke({"context": context, "question": question}).to_string()

# model = genai.GenerativeModel('gemini-1.5-flash')
# response = model.generate_content(prompt)
# print(response.text)