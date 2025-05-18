from Prompts.agent_prompt import C_PROGRAMMING_TEMPLATE
from langchain_core.prompts import ChatPromptTemplate
from Model.models import llm
from Graph.tools.c_programming_tool import get_past_questions

all_tools = [get_past_questions]
def get_c_programming_runnable():
    """
    Returns a runnable object

    Args:
        None

    Returns:
        C Pogramming Runnable
    """

    C_PROGRAMMING_PROMPT = ChatPromptTemplate.from_messages(
        [
            ("system", C_PROGRAMMING_TEMPLATE),
            ("placeholder", "{messages}")
        ]
    )

    c_programming_runnable = C_PROGRAMMING_PROMPT | llm.bind_tools(all_tools)

    return c_programming_runnable