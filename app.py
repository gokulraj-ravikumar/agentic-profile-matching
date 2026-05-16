from __future__ import annotations

from pathlib import Path

import streamlit as st

from matching_agent import run_agent_turn


EXAMPLE_PROMPTS = [
    "Find me candidates with React and 3+ years experience",
    "Compare the top 3 matches side by side",
    "Why did John rank higher than Jane?",
    "Adjust the search to prioritize Python, Airflow, and cloud experience",
]


st.set_page_config(page_title="Agentic Profile Matching", page_icon="🧭", layout="wide")
st.title("Agentic Profile Matching")
st.caption("Chat with the LangGraph agent to rank resumes, compare candidates, and refine requirements mid-conversation.")

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "agent_state" not in st.session_state:
    st.session_state.agent_state = {"messages": [], "screening_round": 0}
if "auto_prompt" not in st.session_state:
    st.session_state.auto_prompt = ""

with st.sidebar:
    st.subheader("Quick Start")
    st.write("Use natural language or paste a JD file path from the workspace.")
    for prompt in EXAMPLE_PROMPTS:
        if st.button(prompt, key=prompt):
            st.session_state.auto_prompt = prompt
    st.divider()
    st.subheader("Workspace Files")
    jd_dir = Path("data/jds")
    if jd_dir.exists():
        jd_files = sorted(path.name for path in jd_dir.glob("*.txt"))
        if jd_files:
            st.write("Sample JD files")
            for file_name in jd_files:
                st.write(f"- {file_name}")
    st.divider()
    st.subheader("Current State")
    agent_state = st.session_state.agent_state
    requirements = agent_state.get("job_requirements", {})
    if requirements:
        st.write(f"Role: {requirements.get('role', 'Not set') or 'Not set'}")
        st.write(f"Must-have: {', '.join(requirements.get('must_have', [])) or 'None yet'}")
        st.write(f"Nice-to-have: {', '.join(requirements.get('nice_to_have', [])) or 'None yet'}")
        st.write(f"Minimum experience: {requirements.get('minimum_experience', 0) or 0}+ years")
    else:
        st.write("No requirements captured yet.")

    shortlist = agent_state.get("candidate_shortlist", [])
    if shortlist:
        st.write("Top matches")
        for candidate in shortlist[:3]:
            st.write(f"- {candidate.get('candidate_name', 'Unknown')} ({candidate.get('match_score', 0)}/100)")
    else:
        st.write("No shortlist yet.")

    if st.button("Clear chat"):
        st.session_state.chat_messages = []
        st.session_state.agent_state = {"messages": [], "screening_round": 0}
        st.session_state.auto_prompt = ""
        st.rerun()

for message in st.session_state.chat_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

auto_prompt = st.session_state.auto_prompt or ""
user_input = st.chat_input("Ask for candidates, compare matches, or refine the search...")
if auto_prompt and not user_input:
    user_input = auto_prompt
    st.session_state.auto_prompt = ""

if user_input:
    st.session_state.chat_messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing resumes and ranking candidates..."):
            updated_state = run_agent_turn(user_input, st.session_state.agent_state)
            st.session_state.agent_state = updated_state
            response_content = str(updated_state.get("final_report", ""))
            st.markdown(response_content)
            st.session_state.chat_messages.append({"role": "assistant", "content": response_content})