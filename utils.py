import streamlit as st
import openai
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

openai.api_key = st.secrets["OPENAI_API_KEY"]

@st.cache_resource
def load_chain(update=False):
    """
    The `load_chain()` function initializes and configures a conversational retrieval chain for
    answering user questions.
    :return: The `load_chain()` function returns a ConversationalRetrievalChain object.
    """
    # if update==False: 
    #     # Load OpenAI embedding model
    embeddings = OpenAIEmbeddings()
        
    # Load OpenAI chat model
    llm = ChatOpenAI(temperature=0)
        
    # Load our local FAISS index as a retriever
    vector_store = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    
    # Create memory 'chat_history' 
    memory = ConversationBufferWindowMemory(k=3,memory_key="chat_history")

    # Create the Conversational Chain
    chain = ConversationalRetrievalChain.from_llm(llm, 
                                                retriever=retriever, 
                                                memory=memory, 
                                                get_chat_history=lambda h : h,
                                                verbose=True)

    # Create system prompt
    template = """
    You are an AI assistant for answering questions about the hugo's minerva experience and notes
    You are given the following extracted parts of a long document and a question. Provide a conversational answer.
    If you are not at least 70 percent sure of your answer, just answer 'Sorry, I don't know... 😔', and then give information about something only if it is 80 percent relevant. 
    Don't try to make up an answer. Don't say something irrelevant please.
    Don't use chat history if it's not related to the query.
    {context}
    Question: {question}
    Helpful Answer:"""

    # Add system prompt to chain
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template)
    chain.combine_docs_chain.llm_chain.prompt.messages[0] = SystemMessagePromptTemplate(prompt=QA_CHAIN_PROMPT)

    return chain

def update_embeddings(docs):
    # Load existing FAISS index if available, otherwise create a new one
    try:
        db1 = FAISS.load_local("faiss_index", OpenAIEmbeddings(),allow_dangerous_deserialization=True)
    except:
        db1 = FAISS.from_documents([], OpenAIEmbeddings())
    
    # Split documents into smaller chunks
    markdown_splitter = RecursiveCharacterTextSplitter(
        separators=["#", "##", "###", "\n\n", "\n", ".", "!", "?"],
        chunk_size=2048,
        chunk_overlap=500)
    
    new_docs = markdown_splitter.split_documents(docs)
    print(new_docs)
    db1.add_documents(new_docs)
    db1.save_local("faiss_index")
    
    print('Local FAISS index has been successfully updated.')