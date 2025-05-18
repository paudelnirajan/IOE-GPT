from core.state import State
from langgraph.prebuilt import tools_condition
from langgraph.graph import END

def agent_router(state: State):
    next_node = tools_condition(state) 
    if next_node == END:
        return END
    ai_message = state["messages"][-1]
    first_tool_call = ai_message.tool_calls[0]
    return first_tool_call["name"]