from dotenv import load_dotenv
load_dotenv()
import os
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.environ["GOOGLE_API_KEY"],
    temperature=0.2,
)

instructor_embeddings = HuggingFaceInstructEmbeddings(
    model_name="hkunlp/instructor-large", 
)

database_path = "faiss_index_vectordb"

def create_vector_db():
    loader = CSVLoader(file_path="D:/CodeBasics_QandA/codebasics_faqs.csv", source_column='prompt')
    data = loader.load()
    vectordb = FAISS.from_documents(documents=data, embedding=instructor_embeddings)
    vectordb.save_local(database_path)

def get_qa_chain():
    vectordb = FAISS.load_local(database_path, instructor_embeddings, allow_dangerous_deserialization=True)
    retriever = vectordb.as_retriever()

    class QAChain:
        def invoke(self, query):
            docs = retriever.get_relevant_documents(query["input"])
            context = "\n\n".join([doc.page_content for doc in docs])
            
            prompt = f"""You are a helpful assistant. Given the following context and question, generate an answer based on this context only. In the answer, try to provide as much text possible from the response section in the source document context without making it yourself. If the answer is not found in the context, kindly state 'I don't know'. Do not try to make up an answer.

CONTEXT: {context}
QUESTION: {query["input"]}

ANSWER:"""
            
            response = llm.invoke(prompt)
            return {"answer": response.content}
    
    return QAChain()