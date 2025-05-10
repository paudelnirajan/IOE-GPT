from langchain_core.prompts import ChatPromptTemplate
from Prompts.agent_prompt import SUPERVISOR_TEMPLATE
from Model.models import llm
from Core.state import ToComputerProgramming

all_tools = [ToComputerProgramming]
def get_supervisor_runnable():
    """Returns a supervisor runnable
    
    Args:
    None

    Returns:
    Runnable Object
    """

    SUPERVISOR_PROMPT = ChatPromptTemplate.from_messages(
        [
            ("system", SUPERVISOR_TEMPLATE),
            ("placeholder", "{messages}")
        ]
    )

    supervisor_runnable = SUPERVISOR_PROMPT | llm.bind_tools(all_tools) 
    return supervisor_runnable

