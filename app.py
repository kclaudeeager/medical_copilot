# Adapted from https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps#build-a-simple-chatbot-gui-with-streaming
import os
import base64
import gc
import tempfile
import uuid

from IPython.display import Markdown, display
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.core import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, ServiceContext, SimpleDirectoryReader

import streamlit as st

# Ensure session state attributes are initialized
if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()  # Generate a unique session ID
if "file_cache" not in st.session_state:
    st.session_state.file_cache = {}  # Initialize file cache
if "messages" not in st.session_state:
    st.session_state.messages = []  # Initialize chat messages
if "context" not in st.session_state:
    st.session_state.context = None 

session_id = st.session_state.id
print("Session: ", session_id)
client = None

def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None
    gc.collect()

def display_pdf(file):
    # Opening file from file path
    st.markdown("### PDF Preview")
    base64_pdf = base64.b64encode(file.read()).decode("utf-8")

    # Embedding PDF in HTML
    pdf_display = f"""<iframe src="data:application/pdf;base64,{base64_pdf}" width="400" height="100%" type="application/pdf"
                        style="height:100vh; width:100%"
                    >
                    </iframe>"""
    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)

with st.sidebar:
    st.header(f"Add your documents!")
    
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

                    file_key = f"{session_id}-{uploaded_file.name}"
                    st.write(f"Indexing your document: {uploaded_file.name}")

                    # Index the file if it hasn't been cached already
                    if file_key not in st.session_state.get('file_cache', {}):

                        if os.path.exists(temp_dir):
                            loader = SimpleDirectoryReader(
                                input_dir=temp_dir,
                                required_exts=[".pdf"],
                                recursive=True
                            )
                        else:
                            st.error('Could not find the file you uploaded, please check again...')
                            st.stop()

                        docs.extend(loader.load_data())
                    else:
                        st.write(f"Document {uploaded_file.name} already cached.")

                # Setup LLM & embedding model
                llm = Ollama(model="llama3:8b-instruct-q4_1", request_timeout=120.0)
                embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5", trust_remote_code=True)

                # Create index over all loaded documents
                Settings.embed_model = embed_model
                index = VectorStoreIndex.from_documents(docs, show_progress=True)

                # Create the query engine, using RAG over indexed documents
                Settings.llm = llm
                query_engine = index.as_query_engine(streaming=True)

                # ====== Customise prompt template with Clinical Assistant Instructions ======
                broad_prompt = '''
                You are an expert clinical assistant AI. Your task is to help healthcare providers with diagnostic support based on patient symptoms, medical history, and local disease data.

                - If the patient's symptoms match a known diagnosis based on your pre-existing knowledge or provided documents, suggest the diagnosis and treatment options.
                - Otherwise, use the search tool to retrieve relevant medical information.
                - In addition to diagnosis, suggest follow-up consultation questions for the healthcare provider to ask the patient to ensure a comprehensive diagnosis.
                - Provide classifications of possible diseases based on the symptoms, including differential diagnoses (similar conditions to rule out).
                - Ensure to ask further clarifying questions that the doctor should consider, such as lifestyle factors, prior medical history, or exposure to local disease trends, to refine the diagnosis.
                - If the user does not provide all the necessary information (e.g., symptoms, medical history, or location), ask them to specify the missing fields.
                '''
                
                qa_prompt_tmpl = PromptTemplate(broad_prompt)

                query_engine.update_prompts(
                    {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
                )
                
                # Cache the query engine
                st.session_state.file_cache[session_id] = query_engine

                # Inform the user that the file is processed and Display the PDF uploaded
                st.success("Ready to Chat!")
                # for file in uploaded_files:
                #     display_pdf(file)

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.stop()     

col1, col2 = st.columns([6, 1])

with col1:
    st.header(f"Chat with Docs using Llama-3.1")

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

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Simulate stream of response with milliseconds delay
        streaming_response = st.session_state.file_cache[session_id].query(prompt)
        
        for chunk in streaming_response.response_gen:
            full_response += chunk
            message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
