from langgraph.graph import MessageGraph, END
from typing import List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from chains import responder_chain, revisor_chain
from tool_executor import execute_tool
from models import ReviseAnswer


RESPONDER = "responder"
REVISOR =  "revisor"
TOOLS_EXECUTOR = "tools_executor"



def responder_node(state: List[BaseMessage]) -> List[BaseMessage]:
    return responder_chain.invoke({"messages": state})


def revisor_node(state: List[BaseMessage]) -> List[BaseMessage]:
    return revisor_chain.invoke({"messages": state})


def should_continue(state: List[BaseMessage]) -> str:
    if len(state) > 3:
        return END
    return REVISOR


# Graph Building.
graph_builder = MessageGraph()

graph_builder.add_node(RESPONDER, responder_node)
graph_builder.add_node(TOOLS_EXECUTOR, execute_tool)
graph_builder.add_node(REVISOR, revisor_node)

graph_builder.set_entry_point(RESPONDER)
graph_builder.add_edge(RESPONDER, TOOLS_EXECUTOR)
graph_builder.add_edge(TOOLS_EXECUTOR, REVISOR)
graph_builder.add_conditional_edges(REVISOR, should_continue, path_map = {RESPONDER: RESPONDER, END: END})

graph = graph_builder.compile()


if __name__ == "__main__":
    
    tweet = graph.invoke(
        [
            HumanMessage(content = "Write a LinkedIn blog post in the topic of **Python for AI Agents**.")
        ]
    )
    print(f"All response contains {len(tweet)} BaseMessage those are:\n{tweet}", end = "\n\n\n")

    last_response: AIMessage = tweet[-1]

    print(f"Final Response: {last_response}", end = "\n\n\n")

    try:
        
        from chains import revisor_chain_validator
        final_structured_output: List[ReviseAnswer] = revisor_chain_validator.invoke(last_response)
        print(final_structured_output[0].model_dump_json(indent = 6))
    
    except Exception as e:
        print(f"Error occur during parse the output. {e}")

    


        
        


    




