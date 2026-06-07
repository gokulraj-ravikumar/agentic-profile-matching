# Agentic Profile Matching 

An advanced AI-powered resume matching application built with **LangGraph**, **LangChain**, and **ChromaDB**. This agent automates the recruitment screening process by taking natural language job descriptions, extracting requirements, and searching through a local repository of resumes to find the best candidates.

## Features

- **Agentic Workflow**: A state-driven LangGraph architecture that automatically parses job descriptions, searches resumes, ranks candidates, and generates a side-by-side comparison report.
- **Multi-Round Screening**: Simulates a multi-round interview process, identifying top candidates, providing a deep-dive analysis, and outputting actionable Hire/No-Hire recommendations.
- **RAG-based Resume Search**: Uses OpenAI Embeddings and ChromaDB to semantically search and match resumes against must-have skills and required experience.
- **Conversational Refinement**: Allows recruiters to iteratively refine job requirements via natural language feedback.
- **Dual Interfaces**: Choose between a lightweight Command Line Interface (CLI) or an interactive Web UI (Streamlit).

## Prerequisites

- Python 3.9+
- OpenAI API Key

## Setup Instructions

1. **Clone or Download the Repository**
   Ensure you are in the project root folder: `agentic-profile-matching`.

2. **Set up the Virtual Environment**
   To avoid dependency conflicts, create and activate a virtual environment:
   ```powershell
   python -m venv venv
   
   # Note: If you get a PowerShell Execution Policy error, run this first:
   # Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
   
   .\venv\Scripts\Activate.ps1
   ```

3. **Install Dependencies**
   Install the required Python packages:
   ```powershell
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**
   Open the `.env` file in the root directory and add your OpenAI API Key:
   ```text
   OPENAI_API_KEY="sk-..."
   ```

## Usage Instructions

Before searching for candidates, the agent needs to index the resumes in `data/resumes/` into a vector database (ChromaDB). 

### Option 1: Command Line Interface (CLI)

1. Activate your virtual environment (if not already activated):
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```
2. Run the CLI app:
   ```powershell
   python cli_app.py
   ```
3. **On your first run:** Select `Option 1` to process the resumes into the Vector DB.
4. **On subsequent runs:** Select `Option 2` to start the conversational agent. Paste a job description or state your requirements (e.g., "Find me QA engineers with Python experience") and interact with the agent!

### Option 2: Streamlit Web UI

1. Activate your virtual environment:
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```
2. Start the Streamlit server:
   ```powershell
   streamlit run streamlit_app.py
   ```
3. **On your first run:** Look at the left sidebar and click **"Process Resumes into Vector DB"**. Wait for the success message.
4. Use the chat interface to paste your job description and converse with the agent. You can ask follow-up questions like "Why did you pick the top candidate?" or refine your search by saying "Actually, they must have 5 years of experience."

## Project Structure

- `matching_agent.py` - Compiles the LangGraph state machine.
- `state.py` - Defines the core memory and variables tracked by the agent.
- `nodes.py` - The operational steps of the agent (parsing, searching, ranking, reporting).
- `tools.py` - LLM functions and database handlers available to the agent.
- `cli_app.py` / `streamlit_app.py` - User interfaces.
- `generate_resumes.py` - A utility script to generate dummy IT resumes.
- `data/resumes/` - The directory containing candidate resumes (PDF, DOCX, TXT).
