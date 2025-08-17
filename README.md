# CodeBasics Q&A ChatBot 🤖

A powerful Question & Answer system built with LangChain that provides intelligent responses to CodeBasics-related queries using Google's Gemini AI model and vector similarity search.

## 📋 Overview

This project implements a Retrieval-Augmented Generation (RAG) system that:
- Loads FAQ data from CSV files
- Creates vector embeddings using HuggingFace's instructor-large model
- Stores vectors in FAISS database for efficient similarity search
- Uses Google's Gemini 1.5 Flash model for intelligent response generation
- Provides a user-friendly Streamlit web interface

## 🚀 Features

- **Intelligent Q&A**: Ask questions and get contextual answers based on CodeBasics content
- **Vector Search**: Efficient similarity search using FAISS vector database
- **Source-Grounded**: Responses are based only on provided context, avoiding hallucinations
- **Easy Setup**: Simple one-click knowledge base creation
- **Interactive UI**: Clean Streamlit interface for seamless user experience

## 🛠️ Technology Stack

- **Framework**: LangChain
- **LLM**: Google Gemini 1.5 Flash
- **Embeddings**: HuggingFace Instructor-Large
- **Vector Database**: FAISS
- **UI**: Streamlit
- **Language**: Python

## 📦 Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd CodeBasics_QandA
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the project root and add your Google API key:
```env
GOOGLE_API_KEY=your_google_api_key_here
```

## 🔧 Dependencies

The project requires the following packages (see `requirements.txt`):

- `langchain==0.0.284` - Main LangChain framework
- `langchain-google-genai==1.0.10` - Google Gemini integration
- `langchain-community==0.0.38` - Community extensions for LangChain
- `python-dotenv==1.0.0` - Environment variable management
- `streamlit==1.22.0` - Web interface framework
- `tiktoken==0.4.0` - Tokenization library
- `faiss-cpu==1.7.4` - Vector similarity search
- `protobuf~=3.19.0` - Protocol buffers
- `google-generativeai==0.3.2` - Google AI SDK
- `sentence-transformers==2.2.2` - Sentence embedding models
- `InstructorEmbedding==1.0.1` - Instructor embedding models

## 🚀 Usage

1. **Start the application**:
```bash
streamlit run main.py
```

2. **Create Knowledge Base**:
   - Click the "Create Knowledge Base" button to process the FAQ data and create vector embeddings
   - This step needs to be done once or when you update your FAQ data

3. **Ask Questions**:
   - Enter your question in the text input field
   - Get intelligent answers based on the CodeBasics knowledge base

## 📁 Project Structure

```
CodeBasics_QandA/
├── main.py                 # Streamlit application entry point
├── langchain_qa.py         # Core Q&A logic and chain setup
├── requirements.txt        # Python dependencies
├── codebasics_faqs.csv    # FAQ data source
├── faiss_index_vectordb/  # Vector database storage (created after first run)
└── .env                   # Environment variables (create this)
```

## 🔍 How It Works

1. **Data Loading**: FAQ data is loaded from `codebasics_faqs.csv` using LangChain's CSV loader
2. **Embedding Generation**: Text is converted to vectors using HuggingFace's instructor-large model
3. **Vector Storage**: Embeddings are stored in FAISS database for fast similarity search
4. **Query Processing**: User questions are embedded and matched against stored vectors
5. **Response Generation**: Relevant context is passed to Gemini model for answer generation

## 🎯 Key Components

### Vector Database Creation (`langchain_qa.py:26`)
```python
def create_vector_db():
    loader = CSVLoader(file_path="D:/CodeBasics_QandA/codebasics_faqs.csv", source_column='prompt')
    data = loader.load()
    vectordb = FAISS.from_documents(documents=data, embedding=instructor_embeddings)
    vectordb.save_local(database_path)
```

### Q&A Chain Setup (`langchain_qa.py:33`)
```python
def get_qa_chain():
    vectordb = FAISS.load_local(database_path, instructor_embeddings, allow_dangerous_deserialization=True)
    retriever = vectordb.as_retriever(score_threshold=0.5)
    # ... chain configuration
```

## 🔒 Environment Setup

Ensure you have a Google API key for Gemini access. Add it to your `.env` file:

```env
GOOGLE_API_KEY=your_actual_api_key_here
```

## 📝 Notes

- The system is designed to provide answers only based on the provided context
- If information is not found in the knowledge base, it will respond with "I don't know"
- Vector database is saved locally for persistent storage
- First-time setup requires creating the knowledge base before asking questions

## 🤝 Contributing

Feel free to contribute to this project by submitting issues or pull requests.

## 📄 License

This project is open source and available under the MIT License.
