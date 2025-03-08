import streamlit as st
from pinecone import Pinecone
from langchain_community.embeddings import SentenceTransformerEmbeddings
from google import genai
import os


PINECONE_API_KEY = 'pcsk_5NBr39_SEsKT2c238kck4UrJw5HE8GmV4qPHuTPesgwm6GTGtaFNgL3Q4dGehsmryNg1ws'
GEMINI_API_KEY = 'AIzaSyCRP0bFr9e3ebbK-J01Fsnf43JiFf3PYuc'


whisper_model = whisper.load_model("base")
pc = Pinecone(api_key=PINECONE_API_KEY)


cp_index = pc.Index('causes-prevention')
dt_index = pc.Index('diagnos-treatment')
is_index = pc.Index('intro-symptoms')


client = genai.Client(api_key=GEMINI_API_KEY)
embeddings = SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")




def generate_response(query: str, chat_history: str) -> str:
    query_embedding = embeddings.embed_query(query)

    cps = cp_index.query(vector=query_embedding, top_k=5, include_metadata=True)
    dts = dt_index.query(vector=query_embedding, top_k=5, include_metadata=True)
    iss = is_index.query(vector=query_embedding, top_k=5, include_metadata=True)
    
    cps_texts = "\n".join([match["metadata"]["text"] for match in cps["matches"]])
    dts_texts = "\n".join([match["metadata"]["text"] for match in dts["matches"]])
    iss_texts = "\n".join([match["metadata"]["text"] for match in iss["matches"]])

    summarizer = client.models.generate_content(
        model="gemini-2.0-flash", 
        contents=f"Act as a doctor, summarize the information for providing useful insights to another doctor using causes, prevention, diagnos,treatments and symptoms.\n\nHere is the basic infor about diseases that might be happening and its symptoms\n{dt_index}  \n\n  Here is the causes and prevention:\n {cps_texts}\n\n Here is information about diagnos and treatment: \n {dts_texts}"
    ) 
    response = client.models.generate_content(
        model='gemini-2.0-flash',
        contents=f"Act as a doctor, provide guidlines and he need of treatment if any to the patient according to the given data about the patient in maximum 200 words: \n\n {summarizer.text}. Here is the past chat history: {chat_history}\n\n Here is the query: {query}"
    )
    return response.text


st.set_page_config(page_title="Medical Bot")

st.markdown(
    "<h1 style='text-align: center;'>🔍 RAG Chatbot with Pinecone & Gemini</h1>",
    unsafe_allow_html=True
)


if "messages" not in st.session_state:
    st.session_state.messages = []


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
