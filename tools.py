import os
import json
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from fs_tools import read_file, list_files, write_file, search_in_file
from job_matcher import JobMatcherEngine
from resume_rag import ResumeRAGPipeline

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0) if os.getenv("OPENAI_API_KEY") else None

@tool
def extract_requirements(jd: str) -> str:
    """Extract must-have and nice-to-have requirements from a job description."""
    if not llm:
        return json.dumps({"must_have": [], "nice_to_have": [], "minimum_experience": 0})
    
    prompt = f"""
    You are an expert technical recruiter. Your task is to extract the explicitly required technical skills and minimum years of experience from the job description or user request.
    
    IMPORTANT: The text may be a conversational request (e.g., "give me candidates for .net profile"). 
    Even if it is conversational, you MUST extract the technical skills mentioned (e.g., ".net").
    If the text includes a "User Update:", that update OVERRIDES previous requirements. Only extract the newly requested skills from the update.
    
    Return ONLY a valid JSON object:
    {{
      "must_have": ["skill1", "skill2"],
      "nice_to_have": ["skill3"],
      "minimum_experience": 0
    }}
    
    Job Description / Request:
    {jd}
    
    Return ONLY the JSON.
    """
    try:
        response = llm.invoke(prompt)
        text = str(response.content).strip()
        import re
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            text = match.group(0)
        return text
    except Exception as e:
        return json.dumps({"error": str(e), "must_have": [], "nice_to_have": [], "minimum_experience": 0})

@tool
def search_resumes(job_description: str, must_have: list, min_exp: int) -> str:
    """Search for candidates matching the job description using RAG."""
    if not os.getenv("OPENAI_API_KEY"):
        return json.dumps({"error": "No OpenAI API Key set."})
        
    engine = JobMatcherEngine()
    results = engine.match_jobs(
        job_description=job_description,
        must_have_skills=must_have,
        minimum_experience=min_exp,
        top_n=10
    )
    return json.dumps(results)

@tool
def compare_candidates(candidate_data_list: list) -> str:
    """Compare multiple candidates side-by-side."""
    if not llm:
        return "LLM not configured."
        
    prompt = f"""
    You are an expert recruiter. Compare the following candidates side-by-side based on their match scores, skills, and experience.
    Provide a clear and concise comparison report highlighting strengths and weaknesses.
    
    Candidates Data:
    {json.dumps(candidate_data_list, indent=2)}
    """
    return llm.invoke(prompt).content

@tool
def generate_interview_questions(candidate_data: dict, job_description: str) -> str:
    """Generate interview screening questions for a specific candidate."""
    if not llm:
        return "LLM not configured."
        
    prompt = f"""
    Based on the candidate's profile and the job description, generate 5 targeted interview screening questions.
    Focus on their weaknesses or gaps, as well as confirming their core strengths.
    
    Candidate Data:
    {json.dumps(candidate_data, indent=2)}
    
    Job Description:
    {job_description}
    """
    return llm.invoke(prompt).content

# Provide access to filesystem tools
@tool
def fs_read_file(filepath: str) -> str:
    """Read a file using filesystem tools."""
    return json.dumps(read_file(filepath))

@tool
def fs_list_files(directory: str) -> str:
    """List files in a directory."""
    return json.dumps(list_files(directory))

@tool
def process_resumes_db() -> str:
    """Trigger the RAG pipeline to process resumes into ChromaDB."""
    try:
        pipeline = ResumeRAGPipeline()
        pipeline.process_resumes()
        return "Successfully processed resumes into vector database."
    except Exception as e:
        return f"Error: {str(e)}"
