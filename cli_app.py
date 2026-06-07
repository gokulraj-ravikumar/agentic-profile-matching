import uuid
from langchain_core.messages import HumanMessage
from matching_agent import agent_app
from tools import process_resumes_db
import os
from dotenv import load_dotenv

load_dotenv()

def main():
    print("Welcome to the Agentic Resume Matcher CLI!")
    print("------------------------------------------")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY is not set in .env. LLM features will fail.")
        
    print("1. Process Resumes into Vector DB")
    print("2. Start Conversational Agent")
    choice = input("Enter your choice (1 or 2): ")
    
    if choice == '1':
        print("\nProcessing resumes...")
        result = process_resumes_db.invoke({})
        print(result)
        return
        
    if choice == '2':
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        
        print("\nAgent initialized. Type 'exit' to quit.")
        print("Start by describing the job requirements or pasting a Job Description.")
        
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in ['exit', 'quit']:
                break
                
            # If this is not the first message, route it through human_feedback
            # We can just send a HumanMessage and let the graph handle it
            
            # For simplicity in this CLI, we will just stream the graph output
            # In a real scenario, we'd check if we need to call human_feedback_node explicitly 
            # or just start from 'parse_jd'
            
            # Getting current state
            state = agent_app.get_state(config)
            
            if not state.values:
                # First run, start from parse_jd
                inputs = {"messages": [HumanMessage(content=user_input)]}
            else:
                # Subsequent runs, we route to human_feedback
                inputs = {"messages": [HumanMessage(content=user_input)]}
                
                # To force LangGraph to run human_feedback as the entry point for this turn
                # we can use the `update_state` or just let the default routing handle it if configured
                # Let's just invoke with the human message. 
                # Wait, our graph currently starts at `parse_jd` unconditionally on START.
                # If we just invoke, it will run from the last blocked node. But our graph goes to END.
                # So next invoke will start at START (parse_jd) again, which is fine because parse_jd_node
                # handles state accumulation. 
            
            print("\nAgent is thinking...")
            for s in agent_app.stream(inputs, config, stream_mode="values"):
                if "messages" in s and s["messages"]:
                    last_message = s["messages"][-1]
                    if last_message.type == "ai":
                        print(f"\nAgent:\n{last_message.content}")

if __name__ == "__main__":
    main()
