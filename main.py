import streamlit as st
from langchain_qa import create_vector_db, get_qa_chain
st.title("Code Basics Q&A ChatBot")
btn= st.button("Create Knowledge Base")
if btn:
    create_vector_db()
question= st.text_input("Question")
if question:
    chain= get_qa_chain()
    response= chain.invoke({"input": question})
    st.header("Answer: ")
    st.write(response["answer"])