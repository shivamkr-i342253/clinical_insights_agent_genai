"""
This node summarizes the conversation history to date and adds it to the state. This allows us to keep important context in the state while removing old messages to stay within token limits.
"""

from langchain_core.messages import HumanMessage, SystemMessage, RemoveMessage
from chains.chains import get_llm
from graph.state import GraphState
 
def conversation_summary_node(state: GraphState):
   
    # First get the summary if it exists
    summary = state.get("conversation_summary", "")
 
    # Create our summarization prompt
    if summary:
       
        # If a summary already exists, add it to the prompt
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
            "Only include the conversation details and not the tool call response."
            "Exclude the clinical trial dataset from the summary."
            "Take into account important details as mentioned in the conversation."
        )
       
    else:
        # If no summary exists, just create a new one
        summary_message = """
            Create a summary of the conversation above. 
            Only include the conversation details and not the tool call response. 
            Exclude the clinical trial dataset from the summary.
            Take into account important details as mentioned in the conversation.
            """
        
    # Add prompt to our history
    llm = get_llm()
    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = llm.invoke(messages)
   
    # Delete all but the 2 most recent messages and add our summary to the state
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"conversation_summary": response.content, "messages": delete_messages}