from Core.state import State
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import ToolNode
from typing import Callable
from langchain_core.runnables import RunnableLambda
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    logger.debug(f"Handling tool error: {error}")
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }
    
def create_tool_node_with_fallback(tools: list) -> dict:
    logger.debug(f"Creating tool node with {len(tools)} tools")
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )

def create_entry_node(assistant_name: str, new_current_state: str) -> Callable:
    def entry_node(state: State) -> dict:
        tool_call_id = state["messages"][-1].tool_calls[0]["id"]
        logger.info(f"Transitioning to assistant: {assistant_name}")
        return {
            "messages": [
                ToolMessage(content=f"You are now the assistant: {assistant_name}. Review the prior conversation. The user's goal is unmet. Use tools as needed to resolve the task. If the user shifts focus or no longer needs this, call CompleteOrEscalate. Stay silent about your role—just act on behalf of the assistant.",
                tool_call_id=tool_call_id)
            ],
            "current_assistant": new_current_state,
        }
    return entry_node

def pop_current_assistant(state: State) -> dict:
    """Pop the current assistant from the stack and return to the primary assistant.

    This lets the full graph explicitly track the dialog flow and delegate control
    to specific sub-graphs.
    """
    messages = []
    if state["messages"][-1].tool_calls:
        messages.append(
            ToolMessage(
                content="Resuming dialog with the host assistant. Please reflect on the past conversation and assist the user as needed.",
                tool_call_id=state["messages"][-1].tool_calls[0]["id"],
            )
        )
    logger.info("Popping current assistant and returning to host")
    return {
        "current_assistant": "pop",
        "messages": messages,
    }