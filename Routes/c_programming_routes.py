from Core.state import State
from Core.assistant import CompleteOrEscalate
from langgraph.prebuilt import tools_condition
from langgraph.graph import END
import logging
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def c_programming_agent_routes(state: State) -> str:
    """
    Route handler for C Programming Agent that determines the next node in the graph.
    
    Args:
        state (State): The current state of the conversation
        
    Returns:
        str: The name of the next node to execute
    """
    try:
        # Log the incoming state
        logger.debug(f"Processing state with {len(state['messages'])} messages")
        
        # Check for tools condition
        route = tools_condition(state)
        logger.debug(f"Tools condition returned: {route}")
        
        if route == END:
            logger.info("Ending conversation as per tools condition")
            return END
        
        # Validate tool calls exist
        if not state["messages"]:
            logger.error("No messages in state")
            return "leaveskill"
            
        last_message = state["messages"][-1]
        if not hasattr(last_message, 'tool_calls'):
            logger.error("Last message has no tool_calls attribute")
            return "leaveskill"
            
        tool_calls = last_message.tool_calls
        logger.debug(f"Tool calls found: {tool_calls}")

        # Check for cancellation
        did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
        if did_cancel:
            logger.info("Cancellation requested, leaving skill")
            return "leaveskill"
        
        # Define available tool routes
        tool_routes: Dict[str, str] = {
            "c_programming_questions": "c_programming_questions",
            # XXX add other tools later...
        }
        
        # Get the first tool name
        if not tool_calls:
            logger.warning("No tool calls found, defaulting to c_programming_questions")
            return "c_programming_questions"
            
        tool_name = tool_calls[0]["name"]
        logger.debug(f"Selected tool: {tool_name}")
        
        # Route to appropriate tool
        if tool_name in tool_routes:
            logger.info(f"Routing to {tool_routes[tool_name]}")
            return tool_routes[tool_name]
        else:
            logger.warning(f"Unknown tool {tool_name}, defaulting to c_programming_questions")
            return "c_programming_questions"
            
    except Exception as e:
        logger.error(f"Error in c_programming_agent_routes: {str(e)}", exc_info=True)
        return "leaveskill"