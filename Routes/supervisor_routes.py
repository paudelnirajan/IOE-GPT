from langgraph.graph import END
from langgraph.prebuilt import tools_condition
from Core.state import State, ToComputerProgramming
import logging
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def supervisor_routes(state: State) -> str:
    """
    Routes from supervisor node to different other agents based on the calls.
    
    Args:
        state (State): The current state of the conversation
        
    Returns:
        str: The name of the next node to execute
    """
    try:
        # Log the incoming state
        logger.debug(f"Processing supervisor state with {len(state.get('messages', []))} messages")
        
        # Check for tools condition
        route = tools_condition(state)
        logger.debug(f"Tools condition returned: {route}")
        
        if route == END:
            logger.info("Ending conversation as per tools condition")
            return END

        # Safely extract tool calls
        tool_calls = []
        if "messages" in state and state["messages"]:
            last_message = state["messages"][-1]
            if hasattr(last_message, "tool_calls"):
                tool_calls = last_message.tool_calls or []
                logger.debug(f"Tool calls found: {tool_calls}")
            else:
                logger.warning("Last message has no tool_calls attribute")
        else:
            logger.warning("No messages in state")

        if tool_calls:
            tool_name = tool_calls[0]["name"]
            logger.debug(f"Selected tool: {tool_name}")

            # Define available tool routes
            tool_routes: Dict[str, str] = {
                ToComputerProgramming.__name__: "enter_c_programming_agent",
                # XXX add other later
            }

            next_node = tool_routes.get(tool_name)
            if next_node:
                logger.info(f"Routing to node: {next_node}")
                return next_node
            else:
                logger.warning(f"Unknown tool call '{tool_name}' from primary_assistant")
                return END

        logger.info("No specific routing decision, continuing with primary_assistant")
        return "primary_assistant"
            
    except Exception as e:
        logger.error(f"Error in supervisor_routes: {str(e)}", exc_info=True)
        return END