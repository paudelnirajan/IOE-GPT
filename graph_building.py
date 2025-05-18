from langgraph.graph import StateGraph
from Graph.tools.c_programming_tool import get_past_questions
from core.assistant import Assistant, create_tool_node_with_fallback
from core.state import State
from Graph.assistants.c_programing_agent import get_c_programming_runnable
from langgraph.graph import START, END 
from Graph.routes.c_programming_router import agent_router
from langgraph.checkpoint.memory import MemorySaver

builder = StateGraph(State)

builder.add_node("c_programming_assistant", Assistant(get_c_programming_runnable()))
builder.add_node("get_past_questions_tool", create_tool_node_with_fallback([get_past_questions]))

builder.add_edge(START, "c_programming_assistant")
builder.add_conditional_edges(
    "c_programming_assistant", 
    agent_router,
    [
        "get_past_questions_tool",
        END
    ]
    )

builder.add_edge("get_past_questions_tool", "c_programming_assistant")

memory = MemorySaver()

graph = builder.compile(checkpointer=memory)

