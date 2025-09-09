import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import os
import logging
import random
import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA

# Importing necessary components of the chatbot
from chatbot_tone import generate_persona, CUSTOM_PROMPT
from user_analysis import analyze_conversation
from ai_analysis import analyze_ai_messages

# Load environment variables
load_dotenv()

# Configure Gemini API for the conversation
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello There! How are you doing today?"}
    ]

# Streamlit app
st.title("Welcome")

# Handling errors in chatbot
logging.basicConfig(filename='chatbot_errors.log', level=logging.ERROR)

# Immitating a Random Persona
if "ai_persona" not in st.session_state:
    st.session_state.ai_persona = generate_persona()

# Function to generate AI response
def generate_response(prompt, chat_history):
    model = genai.GenerativeModel('gemini-2.5-flash')
    ai_persona = st.session_state.ai_persona
    conversation = f"AI Persona: {ai_persona['name']}, Head of {ai_persona['department']}\n\n"
    conversation += prompt + "\n\n" + "\n".join([f"{'User' if m['role'] == 'user' else 'AI'}: {m['content']}" for m in chat_history])

    try:
        response = model.generate_content(conversation)
        # Disabling the autocompletion of the user's response
        ai_response = response.text.strip().split("User:", 1)[0].strip()
        return ai_response
    except Exception as e:
        logging.error(f"Error generating response: {str(e)}")
        return "Oh this seems to be a sensitive topic. What else do you do?"

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
user_input = st.chat_input("Type your message here...")

if user_input:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Display user message
    with st.chat_message("user"):
        st.write(user_input)
    
    ai_response = generate_response(CUSTOM_PROMPT, st.session_state.messages)
    
    # Add AI response to chat history
    st.session_state.messages.append({"role": "assistant", "content": ai_response})
    
    # Display AI response
    with st.chat_message("assistant"):
        st.write(ai_response)


# Function to generate unique conversation id
def get_unique_id():
    return random.randint(100, 999)

# Initialize session state for conversation id if none exists
if 'conversation_id' not in st.session_state:
    st.session_state.conversation_id = get_unique_id()

# Function to initialize RAG system
def initialize_rag():
    async def init_async_rag():
        pdf_path = "Knowledgebase.pdf"
        pdf_loader = PyPDFLoader(pdf_path)
        pages = pdf_loader.load_and_split()
    
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=6000, chunk_overlap=800)
        context = "\n\n".join(str(p.page_content) for p in pages)
        texts = text_splitter.split_text(context)
    
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GOOGLE_API_KEY)
        vector_index = Chroma.from_texts(texts, embeddings).as_retriever(search_kwargs={"k": 5})
    
        model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0.3)
        qa_chain = RetrievalQA.from_chain_type(model, retriever=vector_index, return_source_documents=True)
    
        return qa_chain
    
    return asyncio.run(init_async_rag())

# Session state for RAG system
if "rag_system" not in st.session_state:
    st.session_state.rag_system = initialize_rag()

# Sidebar for additional controls ------------------------------------------------------------------------------
with st.sidebar:
    st.title("Chat Controls")
    if st.button("Clear Chat"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello There! How are you doing today?"}
        ]
        st.rerun()

# Choosing the user's or AI's analysis
    st.subheader("Conversation Analysis")
    analysis_type = st.radio("Choose analysis type:", ["User Susceptibility", "AI Linguistic Tactics"])

# Social Engineering Analysis button
    if st.button("Generate Feedback"):
        if len(st.session_state.messages) > 1 and st.session_state.rag_system:
            conversation = st.session_state.messages
            if analysis_type == "User Susceptibility":
                analysis = analyze_conversation(conversation, st.session_state.rag_system, st.session_state.conversation_id)
            else:
                analysis = analyze_ai_messages(conversation, st.session_state.rag_system, st.session_state.conversation_id)
            st.markdown(analysis)
        else:
            st.warning("Not enough conversation to analyse yet.")