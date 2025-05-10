from typing import Annotated, Literal, Optional
from groq import BaseModel
from pydantic import Field
from typing_extensions import TypedDict
from langgraph.graph.message import AnyMessage, add_messages


def update_dialog_stack(left: list[str], right: Optional[str]) -> list[str]:
    """Push or pop the state."""
    if right is None:
        return left
    if right == "pop":
        return left[:-1]
    return left + [right]


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_info: str
    dialog_state: Annotated[
        list[
            Literal[
                "supervisor",
                "c_prgramming_assistant",
            ]
        ],
        update_dialog_stack,
    ]

class ToComputerProgramming(BaseModel):
    """Transfer work to computer programming agent to handles queries related to subject computer programming. This agent handles the queries about past IOE questions related to computer programing.Some of the topics related to computer programming are 
    "programming_fundamentals", 
    "algorithm_and_flowchart", 
    "introduction_c_programming", 
    "data_and_expressions", 
    "input_output", 
    "control_structures", 
    "arrays_strings_pointers", 
    "functions", 
    "structures", 
    "file_handling", 
    "oop_overview"
    """
    query: str = Field(
        description="user's query about different topics in computer programming"
    )