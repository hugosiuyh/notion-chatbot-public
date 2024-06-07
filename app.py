import os
import time
import streamlit as st
from utils import load_chain, update_embeddings
from langchain_community.document_loaders import PyPDFLoader
import tempfile

# Custom image for the app icon and the assistant's avatar
company_logo = os.path.join(os.path.dirname(__file__), 'logo.png')


# Configure streamlit page
st.set_page_config(
    page_title="Hugo's Notion Chatbot",
    page_icon=company_logo
)
# Display the heading
st.title("Your Notion Chatbot")

# Initialize LLM chain in session_state
if 'chain' not in st.session_state:
    st.session_state['chain'] = load_chain()

# Initialize chat history
if 'messages' not in st.session_state:
    # Start with first message from assistant
    st.session_state['messages'] = [{"role": "assistant", 
                                  "content": "Hi Hugo! I am your personal Chatgbt. How can I help you today?"}]

# Display chat messages from history on app rerun
# Custom avatar for the assistant, default avatar for user
for message in st.session_state.messages:
    if message["role"] == 'assistant':
        with st.chat_message(message["role"], avatar=company_logo):
            st.markdown(message["content"])
    else:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Document upload section
st.sidebar.header("Upload Documents")
uploaded_files = st.sidebar.file_uploader("Upload your documents here", accept_multiple_files=True)

if uploaded_files:
    st.sidebar.write("Processing uploaded documents...")
    for uploaded_file in uploaded_files:
        try:
            if uploaded_file.type == "application/pdf":
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_file_path = tmp_file.name
                    
                    loader = PyPDFLoader(tmp_file_path)
                    documents = loader.load()
                    update_embeddings(documents)
                    st.sidebar.success("Documents uploaded and processed successfully!")
                    st.session_state['chain'] = load_chain(update=True)
                      
                # Clear the uploaded files
                st.sidebar.empty()  # Clears the document upload UI
                st.session_state.uploaded_files = []  # Reset the uploaded files in session state
            else:
                st.sidebar.error(f"Please upload a PDF file. The file {uploaded_file.name} is not a PDF.")
        except Exception as e:
            st.sidebar.error(f"Error processing file {uploaded_file.name}: {e}")
    
# Chat logic
if query := st.chat_input("Ask me anything"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant", avatar=company_logo):
        message_placeholder = st.empty()
        # Send user's question to our chain
        result = st.session_state['chain']({"question": query})
        response = result['answer']
        full_response = ""

        # Simulate stream of response with milliseconds delay
        for chunk in response.split():
            full_response += chunk + " "
            time.sleep(0.05)
            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

    # Add assistant message to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})