import streamlit as st
from pinecone import Pinecone
from langchain_community.embeddings import SentenceTransformerEmbeddings
from google import genai
import os

# Configure API Keys
PINECONE_API_KEY = 'pcsk_5NBr39_SEsKT2c238kck4UrJw5HE8GmV4qPHuTPesgwm6GTGtaFNgL3Q4dGehsmryNg1ws'
PINECONE_INDEX = "pubmedbert-base-embedding1"
GEMINI_API_KEY = 'AIzaSyCRP0bFr9e3ebbK-J01Fsnf43JiFf3PYuc'



# Initialize Pinecone
pc = Pinecone(
    api_key=PINECONE_API_KEY
)
index = pc.Index(PINECONE_INDEX)

# Initialize Gemini
client = genai.Client(api_key=GEMINI_API_KEY)
embeddings = SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")


def retrieve_context(query):
    """Retrieve relevant documents from Pinecone."""
    query_embedding = embeddings.embed_query(query)
    results = index.query(vector=query_embedding, top_k=5, include_metadata=True)
    
    return [match["metadata"]["text"] for match in results["matches"]]
    


def generate_response(query, chat_history):
    """Generate a response using Gemini with retrieved context."""
    retrieved_texts = retrieve_context(query)
    context = "\n".join(retrieved_texts)

    response = client.models.generate_content(
        model="gemini-2.0-flash", 
        contents=f"Use the following context to answer:\n{context}\n\nChat history: {chat_history}\n\nHere is the query:\n{query}"
    )
    
    return response.text


# Streamlit UI
st.set_page_config(page_title="Medical Bot", layout="wide")
st.markdown(
    "<h1 style='text-align: center;'>🔍 RAG Chatbot with Pinecone & Gemini</h1>",
    unsafe_allow_html=True
)


if "messages" not in st.session_state:
        st.session_state.messages = []

# Store chat history

# User input
user_input = st.chat_input("Ask me anything...", key="user_input")
if user_input:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Generate response
    with st.spinner("Thinking..."):
        chat_history = "\n".join(f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages)
        ai_response = generate_response(user_input, chat_history)
        st.session_state.messages.append({"role": "assistant", "content": ai_response})

    # Display AI response
    with st.container():
        for msg in st.session_state.messages:
            if msg['role'] == 'assistant':
                with st.chat_message("assistant"):
                    st.markdown(msg['content'])
            if msg['role'] == 'user':
                with st.chat_message("user"):
                    st.markdown(msg['content'])
