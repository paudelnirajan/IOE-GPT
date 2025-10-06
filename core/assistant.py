from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from .state import State
from langchain_core.messages import ToolMessage, AIMessage
from langgraph.prebuilt import ToolNode
from langchain_core.messages.utils import count_tokens_approximately
from langmem.short_term import SummarizationNode
from Model.models import llm

class Assistant:
    def __init__(self, runnable: Runnable):
        print(f"[INFO] Initializing Assistant with runnable")
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        print(f"[INFO] Assistant called with state and config")
        while True:
            print(f"[INFO] Invoking runnable with state")
            result = self.runnable.invoke(state)
            
            # If the LLM happens to return an empty response, we will re-prompt it
            # for an actual response.
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                print(f"[INFO] Empty response received, re-prompting for real output")
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                print(f"[INFO] Valid response received with {len(result.tool_calls) if result.tool_calls else 0} tool calls")
                
                # If we have tool calls, we should return them for processing by the tool node
                if result.tool_calls:
                    print(f"[INFO] Tool calls found, passing to tool nodes for execution")
                    break
                
                # If we don't have tool calls, check if the content contains function call syntax
                # and process it directly here (this handles cases where the model outputs function call format as text)
                if result.content and "<function=" in result.content:
                    # Replace the response with an error message for now
                    print(f"[INFO] Found function call in text content, replacing with error message")
                    result = AIMessage(content="I need to search for relevant questions. Let me try again with the proper format.")
                    # Continue the loop to retry
                    messages = state["messages"] + [result]
                    state = {**state, "messages": messages}
                    continue
                    
                break
                
        return {"messages": result}
    



def handle_tool_error(state) -> dict:
    print(f"[ERROR] Handling tool error in state")
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    print(f"[ERROR] Error details: {repr(error)}")
    print(f"[ERROR] Number of tool calls: {len(tool_calls)}")
    
    error_messages = {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }
    print(f"[INFO] Generated {len(error_messages['messages'])} error messages")
    return error_messages


def create_tool_node_with_fallback(tools: list) -> dict:
    print(f"[INFO] Creating tool node with {len(tools)} tools")
    tool_node = ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )
    print(f"[INFO] Tool node created successfully with fallback handler")
    return tool_node



def create_summarization_node():
    summarization_model = llm.bind(max_tokens=128)
    
    return SummarizationNode(
        token_counter=count_tokens_approximately,
        model=summarization_model,
        max_tokens=1024,              
        max_tokens_before_summary=512,
        max_summary_tokens=256,      
    )