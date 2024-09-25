import os
import uuid
import threading
import openai
import streamlit as st

from langchain_openai import OpenAIEmbeddings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import gc
import tempfile
# Ensure session state attributes are initialized
if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()  # Generate a unique session ID
if "file_cache" not in st.session_state:
    st.session_state.file_cache = {}  # Initialize file cache
if "messages" not in st.session_state:
    st.session_state.messages = []  # Initialize chat messages
if "pre_existing_index" not in st.session_state:
    st.session_state.pre_existing_index = None  # Index for pre-existing documents

# Set up the OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")

# Create an instance of OpenAIEmbeddings
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Define the broad prompt
broad_prompt = '''
You are an expert clinical assistant AI. Your task is to help healthcare providers with diagnostic support based on patient symptoms, medical history, and local disease data.

- Prioritize using the information provided in the documents.
- If the patient's symptoms match a known diagnosis based on the provided documents, suggest the diagnosis and treatment options.
- Otherwise, use the search tool to retrieve relevant medical information and explicitly state that the information is from an external source.
- In addition to diagnosis, suggest follow-up consultation questions for the healthcare provider to ask the patient to ensure a comprehensive diagnosis.
- Provide classifications of possible diseases based on the symptoms, including differential diagnoses (similar conditions to rule out).
- Ensure to ask further clarifying questions that the doctor should consider, such as lifestyle factors, prior medical history, or exposure to local disease trends, to refine the diagnosis.
- If the user does not provide all the necessary information (e.g., symptoms, medical history, or location), ask them to specify the missing fields.
'''

# Function to reset chat and clear context
def reset_chat():
    st.session_state.messages = []
    gc.collect()

# Function to query OpenAI's chat model (e.g., GPT-4)
def query_openai(prompt, context, model="gpt-4"):
    # Construct the full prompt with context
    full_prompt = f"{broad_prompt}\n\nContext:\n{context}\n\nUser Query: {prompt}"

    response = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": broad_prompt},
            {"role": "user", "content": full_prompt},
        ],
        max_tokens=1500,
        temperature=0.7,
        stream=True  # Enable streaming for real-time response
    )
    return response

# Function to load and index pre-existing books
def load_pre_existing_books():
    documents = []
    documents_folder = "documents"
    
    if os.path.exists(documents_folder):
        for filename in os.listdir(documents_folder):
            if filename.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(documents_folder, filename))
                documents.extend(loader.load())
    
    return documents

# Load and index pre-existing books in a separate thread
def load_and_index_books(progress_bar):
    pre_existing_docs = load_pre_existing_books()
    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(pre_existing_docs)
    pre_existing_vector_store = FAISS.from_documents(documents, embeddings)
    
    # Store the pre-existing vector store directly in session state
    st.session_state.pre_existing_index = pre_existing_vector_store
    st.success("Pre-existing documents loaded and indexed successfully.")
    progress_bar.progress(100)  # Update progress to 100%

# Start the thread to load and index pre-existing books
progress_bar = st.progress(0)
thread = threading.Thread(target=load_and_index_books, args=(progress_bar,))
thread.start()

# Function to retrieve relevant documents from FAISS (RAG)
def retrieve_relevant_docs(prompt):
    index = st.session_state.pre_existing_index
    if index:
        return index.similarity_search(prompt, k=5)  # Get top 5 relevant documents
    return []

with st.sidebar:
    st.header("Add your documents!")

    # Allow multiple PDF uploads
    uploaded_files = st.file_uploader("Choose your `.pdf` files", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                docs = []

                for uploaded_file in uploaded_files:
                    file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getvalue())

                    st.write(f"Indexing your document: {uploaded_file.name}")

                    loader = SimpleDirectoryReader(
                        input_dir=temp_dir,
                        required_exts=[".pdf"],
                        recursive=True
                    )
                    docs.extend(loader.load_data())
                
                # Ensure document_texts is not empty
                if not docs:
                    st.error("No valid documents found. Please ensure the uploaded files are in the correct format.")
                    st.stop()
                st.write("Document text extraction complete.")
                
                # Create FAISS vector store
                vector_store = FAISS.from_documents(docs, embeddings)
                st.session_state.file_cache[st.session_state.id] = vector_store

                st.success("Documents indexed successfully!")

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.stop()

col1, col2 = st.columns([6, 1])

with col1:
    st.header(f"Chat with Docs using GPT-4")

with col2:
    st.button("Clear ↺", on_click=reset_chat)

# Initialize chat history
if "messages" not in st.session_state:
    reset_chat()

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What's up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Retrieve relevant documents from the index if available
    retrieved_docs = retrieve_relevant_docs(prompt)

    # Combine the content of the retrieved documents
    context = "\n".join([doc.page_content for doc in retrieved_docs]) if retrieved_docs else ""

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # Query OpenAI GPT model for a streaming response
        streaming_response = query_openai(prompt, context)

        # Process streaming response
        for chunk in streaming_response:
            content = getattr(chunk.choices[0].delta, 'content', None)
            if content:
                full_response += content
                message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
