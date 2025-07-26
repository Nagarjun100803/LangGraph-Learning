from datetime import datetime
from langchain_core.tools import tool 
from langchain_tavily import TavilySearch
from config import settings
from typing import List, Any
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

class InputSchema(BaseModel):
    """
    InputSchema defines the input structure for numerical operations, containing two fields:
    - num_1: The first number (integer or float).
    - num_2: The second number (integer or float).
    """
    num_1: int | float = Field(description = "The first number (integer or float) eg. 8 or 8.5")
    num_2: int | float = Field(description = "The second number (integer or float) eg. 8 or 8.5")


@tool(
    name_or_callable = "date_time", 
    description = "Use when you need to get current date and time.",
)
def get_current_date():
    """Returns the current Date."""
    return datetime.now().strftime("%Y-%m-%d %I:%M:%S, %p")



@tool(
    name_or_callable = "addition",
    description = "Use when you need to perform addition operation.",
    args_schema = InputSchema
)
def addition(num_1: int | float, num_2: int | float) -> float:
    """Returns the addition of two numbers."""
    return num_1 + num_2



@tool(
    name_or_callable = "multiplication",
    description = "Use when you need to perform multiplication operation.",
    args_schema = InputSchema
)
def multiplication(num_1: int | float, num_2: int | float) -> float:
    """Returns the multiplication of two numbers."""
    return num_1 * num_2



@tool(
    name_or_callable = "subtraction",
    description =  "Use when you need to perform subtraction operation.",
    args_schema = InputSchema
)
def subtraction(num_1: int | float, num_2: int | float) -> float:
    """Returns the subtraction of two numbers."""
    return num_1 - num_2


@tool(
    name_or_callable = "division",
    description = "Use when you need to perform division operation.",
    args_schema = InputSchema
)

def division(num_1: int | float, num_2: int | float) -> float:
    """Returns the division of two numbers."""
    if num_2 == 0:
        raise ValueError("Division by zero is not allowed.")
    return num_1 / num_2



tavily_search = TavilySearch(
    tavily_api_key = settings.tavily_api_key,
    max_results = 2
)

available_tools: List[BaseTool | TavilySearch] = [
    get_current_date, addition, multiplication, subtraction, division, tavily_search
]




if __name__ == "__main__":
    

    output_1: Any = addition.invoke(input = {"num_1": 8, "num_2": 2.9})
    print(output_1, end = "\n")
    
    output_2: Any = get_current_date.invoke(input = "")
    print(output_2, end = "\n")