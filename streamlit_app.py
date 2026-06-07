import streamlit as st
import uuid
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from matching_agent import agent_app
from tools import process_resumes_db

load_dotenv()

st.set_page_config(page_title="Agentic Profile Matching", layout="wide")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

st.title("Agentic Resume Matcher")

# Sidebar for controls
with st.sidebar:
    st.header("Settings & Tools")
    if st.button("Process Resumes into Vector DB"):
        with st.spinner("Processing resumes..."):
            result = process_resumes_db.invoke({})
            st.success(result)
            
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY is not set in .env")

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": "Welcome! Please describe the job requirements or paste a Job Description."})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Enter job description or feedback..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Agent is thinking..."):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            inputs = {"messages": [HumanMessage(content=prompt)]}
            
            # Run the agent workflow
            final_response = ""
            for s in agent_app.stream(inputs, config, stream_mode="values"):
                if "messages" in s and s["messages"]:
                    last_message = s["messages"][-1]
                    if last_message.type == "ai":
                        final_response = last_message.content
            
            if final_response:
                st.markdown(final_response)
                st.session_state.messages.append({"role": "assistant", "content": final_response})
