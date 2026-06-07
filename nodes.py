import json
from langchain_core.messages import AIMessage, HumanMessage
from state import AgentState
from tools import (
    extract_requirements,
    search_resumes,
    compare_candidates,
    generate_interview_questions
)

def parse_jd_node(state: AgentState):
    # Simply ensure JD is in state or extract it from latest human message if missing
    jd = state.get("job_description", "")
    if not jd and state.get("messages"):
        last_msg = state["messages"][-1].content
        # Treat the first message as the JD
        jd = last_msg
    return {"job_description": jd}

def extract_requirements_node(state: AgentState):
    jd = state.get("job_description", "")
    reqs_json_str = extract_requirements.invoke(jd)
    try:
        reqs = json.loads(reqs_json_str)
    except:
        reqs = {"must_have": [], "nice_to_have": [], "minimum_experience": 0}
    
    return {"requirements": reqs, "current_stage": "requirements_extracted"}

def search_resumes_node(state: AgentState):
    jd = state.get("job_description", "")
    reqs = state.get("requirements", {})
    must_have = reqs.get("must_have", [])
    min_exp = reqs.get("minimum_experience", 0)
    
    results_str = search_resumes.invoke({
        "job_description": jd, 
        "must_have": must_have, 
        "min_exp": min_exp
    })
    
    try:
        results = json.loads(results_str)
        candidates = results.get("top_matches", [])
    except:
        candidates = []
        
    return {"candidate_pool": candidates, "current_stage": "initial_screen_done"}

def rank_candidates_node(state: AgentState):
    candidates = state.get("candidate_pool", [])
    # Second round: Deep analysis
    # Here we just take the top 5 for deep analysis to simulate multi-round
    top_candidates = candidates[:5]
    
    # Let's say we refine the ranking based on the requirements explicitly
    # JobMatcherEngine already sorted them by match_score. 
    # We can assign hire/no-hire recommendations
    for cand in top_candidates:
        score = cand.get("match_score", 0)
        if score > 80:
            cand["recommendation"] = "Hire"
        elif score > 60:
            cand["recommendation"] = "Borderline - Needs Deep Interview"
        else:
            cand["recommendation"] = "No-Hire"
            
    return {"shortlist": top_candidates, "current_stage": "deep_analysis_done"}

def generate_report_node(state: AgentState):
    shortlist = state.get("shortlist", [])
    
    if not shortlist:
        report = "No candidates met the requirements."
    else:
        # Generate side-by-side comparison for top 3
        comparison = compare_candidates.invoke({"candidate_data_list": shortlist[:3]})
        
        # Optionally generate interview questions for the top candidate
        top_cand_q = generate_interview_questions.invoke({
            "candidate_data": shortlist[0],
            "job_description": state.get("job_description", "")
        }) if shortlist else ""
        
        report = f"## Candidate Comparison Report\n\n{comparison}\n\n## Top Candidate Screening Questions\n\n{top_cand_q}"
        
    return {"messages": [AIMessage(content=report)], "current_stage": "report_generated"}

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def human_feedback_node(state: AgentState):
    # This node uses the LLM to classify if the user wants to update requirements or just ask a question
    last_msg = state["messages"][-1].content
    
    prompt = f"""
    You are a routing assistant. The user sent a message. You must classify it as 'QUESTION' or 'UPDATE_REQUIREMENTS'.
    
    Rules:
    - If the user is asking "why", "how", "compare", "who", "which", or asking a clarifying question about the currently displayed candidates, return 'QUESTION'.
    - If the user is asking to "find", "search", "suggest", "give me candidates", or providing new job requirements like "java", ".net", "5 years experience", return 'UPDATE_REQUIREMENTS'.
    
    If the user's message is asking to see new candidates for a new profile, it is ALWAYS 'UPDATE_REQUIREMENTS'.
    
    User message: "{last_msg}"
    
    Output ONLY the exact string 'QUESTION' or 'UPDATE_REQUIREMENTS'. Do not add any punctuation or extra words.
    """
    try:
        intent = llm.invoke(prompt).content.strip().upper()
    except Exception:
        intent = "QUESTION"
    
    if intent == "UPDATE_REQUIREMENTS":
        updated_jd = state.get("job_description", "") + "\n\nUser Update: " + last_msg
        return {"job_description": updated_jd, "current_stage": "update_requirements"}
    else:
        return {"current_stage": "answer_question"}

def answer_question_node(state: AgentState):
    last_msg = state["messages"][-1].content
    shortlist = state.get("shortlist", [])
    pool = state.get("candidate_pool", [])
    jd = state.get("job_description", "")
    
    prompt = f"""
    You are an intelligent HR assistant. The user asked a follow-up question: "{last_msg}"
    
    Original Job Description Context: {jd}
    
    Current Candidate Pool Data (for filtering questions): 
    {json.dumps([{ 'name': c.get('candidate_name'), 'experience': c.get('metadata', {}).get('years_of_experience', 0), 'skills': c.get('matched_skills', []) + c.get('metadata', {}).get('skills', []) } for c in pool[:20]], indent=2)}
    
    Current Shortlist Detailed Info (for comparison/explanation questions): 
    {json.dumps(shortlist, indent=2)}
    
    Provide a direct, conversational, and helpful answer to the user's question based on the candidate data provided above. DO NOT generate a full template report. DO NOT generate interview questions unless asked. Just answer the user's question naturally as an AI assistant.
    """
    try:
        response = llm.invoke(prompt)
        content = response.content
    except Exception as e:
        content = f"Error generating answer: {str(e)}"
        
    return {"messages": [AIMessage(content=content)], "current_stage": "question_answered"}
