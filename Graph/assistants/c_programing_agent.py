
from Prompts.agent_prompt import C_PROGRAMMING_TEMPLATE
from langchain_core.prompts import ChatPromptTemplate
from Model.models import llm
from tools.c_programming_tool import c_programming_questions

all_tools = [c_programming_questions]
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
            ("system", C_PROGRAMMING_PROMPT),
            ("placeholder", "{messages}")
        ]
    )

    c_programming_runnable = C_PROGRAMMING_PROMPT | llm.bind_tools(all_tools)

    return c_programming_runnable