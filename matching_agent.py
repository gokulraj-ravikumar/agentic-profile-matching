from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from state import AgentState
from nodes import (
    parse_jd_node,
    extract_requirements_node,
    search_resumes_node,
    rank_candidates_node,
    generate_report_node,
    human_feedback_node,
    answer_question_node
)

def route_start(state: AgentState):
    """Route to parse_jd on first run, or human_feedback on subsequent runs."""
    if state.get("job_description"):
        return "human_feedback"
    return "parse_jd"

def route_feedback(state: AgentState):
    if state.get("current_stage") == "update_requirements":
        return "extract_requirements"
    return "answer_question"

def create_agent():
    workflow = StateGraph(AgentState)

    # Add Nodes
    workflow.add_node("parse_jd", parse_jd_node)
    workflow.add_node("extract_requirements", extract_requirements_node)
    workflow.add_node("search_resumes", search_resumes_node)
    workflow.add_node("rank_candidates", rank_candidates_node)
    workflow.add_node("generate_report", generate_report_node)
    workflow.add_node("human_feedback", human_feedback_node)
    workflow.add_node("answer_question", answer_question_node)

    # Define Edges
    workflow.add_conditional_edges(START, route_start)
    workflow.add_edge("parse_jd", "extract_requirements")
    workflow.add_edge("extract_requirements", "search_resumes")
    workflow.add_edge("search_resumes", "rank_candidates")
    workflow.add_edge("rank_candidates", "generate_report")
    
    # Conditional logic can be implemented manually by caller or via the graph
    # For a purely conversational interface, after generate_report, it goes to END.
    # The next HumanMessage triggers human_feedback
    workflow.add_edge("generate_report", END)
    
    workflow.add_conditional_edges("human_feedback", route_feedback)
    workflow.add_edge("answer_question", END)

    # Compile with memory
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    return app

agent_app = create_agent()
