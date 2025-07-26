from altair import Dict
from agent import agent
from langgraph.graph import StateGraph, END
import operator
from typing import Any, List, Tuple, TypedDict, Annotated
from langchain_core.agents import AgentAction, AgentFinish
import json 
from tools import available_tools
from langchain_core.tools import BaseTool
from dotenv import load_dotenv
from langchain_core.exceptions import OutputParserException
from langchain_tavily import  TavilySearch



load_dotenv() # To make sure all the env variables to Trace using LangSmith.



class AgentState(TypedDict):
    """Represents the state of an agent during execution.
    Attributes:
        input: The input string provided to the agent.
        agent_outcome: The outcome of the agent's action, which can be an AgentAction or AgentFinish.
        intermediate_steps: A list of tuples containing the agent's actions and their corresponding outputs.
    """

    input: str 
    agent_outcome: AgentAction | AgentFinish | None 
    intermediate_steps: Annotated[List[Tuple[AgentAction, str]], operator.concat]



REASON = "reason"
ACTION = "action"


def reason_node(state: AgentState) -> AgentState:
    
    try:
        
        agent_outcome: AgentAction | AgentFinish = agent.invoke(state)
    
    except OutputParserException as e:
        print(f"ReActSingleInputOutputParseException occurs, so type cast the llm output into AgentFinish object", end = "\n")
        agent_outcome: AgentFinish = AgentFinish(
            return_values = {"output": e.llm_output},
            log = "Now, I Know the final answer."
        )

    return {
        "agent_outcome": agent_outcome
    }



def action_node(state: AgentState) -> AgentState:

    agent_action: AgentAction = state["agent_outcome"]
    print(f"Current agent_action: {agent_action}", end = "\n")

    # Extract tool name and tool input from AgentAction
    tool_name: str = agent_action.tool
    tool_input: str = agent_action.tool_input

    tools_by_name: Dict[str, [BaseTool | TavilySearch]] = {tool.name: tool for tool in available_tools}
    # Find the matching tool function.
    tool_function: BaseTool | None = tools_by_name.get(tool_name)
    print(f"Current tool name: {tool_name}", end = "\n")


    # Try to load the tool input, if it is like a JSON.
    parsed_input = tool_input
    if isinstance(parsed_input, str):
        try:
            parsed_input = json.loads(tool_input)
            print(f"The parsed input is {parsed_input}")
        except json.JSONDecodeError:
            pass # Leave it as string if not a valid JSON.

    
    # Execute the tool function with parsed_input. 
    if tool_function:
        try:
            output: Any = tool_function.invoke(input = parsed_input)
        except Exception as e:
            print(f"Unexpected Error occurs {str(e)}")
            output = f"Error occurred while executing tool {tool_name}: {str(e)}"
    else:
        output = f"Tool {tool_name} not found."
    
    print(f"The output of this tool is: {output}", end = "\n")
    return {
        "intermediate_steps": [(agent_action, str(output))]
    }


def should_continue(state: AgentState) -> AgentState:
    
    # print(f"Current agent outcome in should_continue: {state["agent_outcome"]}")
    if isinstance(state["agent_outcome"], AgentFinish):
        return END 
    return ACTION


# Build the state graph for the agent.
graph_builder = StateGraph(AgentState)
graph_builder.add_node(REASON, reason_node)
graph_builder.add_node(ACTION, action_node)

graph_builder.set_entry_point(REASON)
graph_builder.add_edge(ACTION, REASON)
graph_builder.add_conditional_edges(
    REASON,
    should_continue,
    path_map = {
        ACTION: ACTION,
        END: END
    }
)

graph = graph_builder.compile()


if __name__ == "__main__":

    sample_questions: List[str] = [
        "Hello, This is Nagarjun, I'm an AI Engineer",
        # "What is 2.89762 plus 98.872634",
        # "what is 2.89762 plus 98.872634, add the result with 87.8, and muultiply it with 2.5",
        "Get the current date and month and give me their sum and product",
        # "Why langgraph is best in AI Frameworks."
    ]

    
    for sample_question in sample_questions:
        response: AgentState = graph.invoke(
            AgentState(
                input = sample_question,
                agent_outcome = None,
                intermediate_steps = []
            )
        )

        print(f"Question: {sample_question}\n\nAnswer: {response["agent_outcome"]}\n\nIntermediate Steps: {response["intermediate_steps"]}")