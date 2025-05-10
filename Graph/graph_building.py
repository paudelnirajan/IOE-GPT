from Core.assistant import Assistant
from Core.state import State
from langgraph.graph import StateGraph, START, END
from Core.utility import create_entry_node, create_tool_node_with_fallback, pop_current_assistant
from Graph.assistants.c_programing_agent import get_c_programming_runnable
from Graph.assistants.supervisor import get_supervisor_runnable
from Graph.tools.c_programming_tool import c_programming_questions
from Routes.supervisor_routes import supervisor_routes
from Routes.c_programming_routes import c_programming_agent_routes

builder = StateGraph(State)

# assistant nodes
builder.add_node("supervisor", Assistant(get_supervisor_runnable()))
builder.add_node("c_programming_agent", Assistant(get_c_programming_runnable()))

# entry nodes
builder.add_node("enter_c_programming_agent", create_entry_node("Computer Programming Agent", "c_programming_agent"))
builder.add_node("leaveskill", pop_current_assistant)

# tool nodes
builder.add_node("c_programming_questions", create_tool_node_with_fallback(c_programming_questions))

# edges
builder.add_edge(START, "supervisor")
builder.add_conditional_edges(
    "supervisor",
    supervisor_routes,
    {
        "enter_c_programming_agent",
        # XXX add other later,
        END
    }
)

builder.add_conditional_edges(
    "c_programming_agent",
    c_programming_agent_routes,
    {
        "c_programming_questions",
        "leaveskill"
    }
)

# Enter edges
builder.add_edge("enter_c_programming_agent", "c_programming_agent")

# leave skill edges
builder.add_edge("leaveskill", "supervisor")

# Add edge from tool back to agent
builder.add_edge("c_programming_questions", "c_programming_agent")
