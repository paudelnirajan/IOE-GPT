from typing import Literal, Optional, List, Tuple 
from pydantic import BaseModel, Field

class QuestionSearch(BaseModel):
    """Search over the json file about the question of particular year or some particular metadata... ."""

    id: Optional[str] = Field(
        None, # Default to None for optional fields
        description="ID of a particular question. Will hold a value like 'subject_code+question_number'."
    )

    subject: Literal["computer programming"] = Field(
        description="Subject the question belongs to."
    )

    # Modified to accept list of years
    year_ad: Optional[List[int]] = Field(
        None,
        description="List of years in AD that the questions appeared."
    )

    year_bs: Optional[List[int]] = Field(
        None,
        description="List of years in BS that the questions appeared."
    )

    question_text: Optional[str] = Field( # Added field for searching question text
        None,
        description="Keywords or full text to search within the question itself."
    )

    type: Optional[Literal["theory", "programming"]] = Field( 
        None,
        description="Type of the question." 
    )

    format: Optional[Literal["short", "long"]] = Field(
        None,
        description="Format of the question (e.g., short answer, long answer)." 
    )

    marks: Optional[int] = Field(
        None,
        description="Marks allocated to the question being searched."
    )

    topic: Optional[Literal["programming_fundamentals", "algorithm_and_flowchart", "introduction_c_programming", "data_and_expressions", "input_output", "control_structures", "arrays_strings_pointers", "functions", "structures", "file_handling", "oop_overview"]] = Field(
        None,
        description="Topic that the question is from."
    )

    unit: Optional[int] = Field(
        None,
        description="Unit the question is from."
    )

    # tags: Optional[List[str]] = Field(
    #     None,
    #     description="Tags associated with the question."
    # )

    question_number: Optional[str] = Field(
        None,
        description="Question number (e.g., '1a', '2b', 4, 5)."
    )

    source: Optional[Literal["regular", "back"]] = Field(
        None,
        description="Source exam type (regular or back paper)." 
    )

    semester: Optional[Literal["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth"]] = Field(
         None,
         description="Semester the question is for." 
    )

