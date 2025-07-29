from config import settings
from tools import available_tools
from typing import Dict, TypedDict, List, Annotated
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, END, add_messages
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_groq.chat_models import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from dotenv import load_dotenv

load_dotenv()




google_llm = ChatGoogleGenerativeAI(
    google_api_key = settings.google_api_key,
    model = "gemini-2.5-flash",
    max_tokens = 3000,
    temperature = 0.1
)


groq_llm = ChatGroq(
    groq_api_key = settings.groq_api_key,
    temperature = 0.2,
    max_retries = 2,
    max_tokens = 3000,
    model = "llama-3.1-8b-instant"
)

llm_with_tools = groq_llm.bind_tools(tools = available_tools)


class ChatBotState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


def chatbot_node(state: ChatBotState) -> ChatBotState:
    return {
        "messages": [llm_with_tools.invoke(state["messages"])]
    }


def tool_call_required(state: ChatBotState) -> ChatBotState:
    """Check if the last message requires tool calls."""
    last_ai_message: AIMessage = state["messages"][-1]
    if last_ai_message.tool_calls:
        return TOOL
    return END 


CHATBOT = "chatbot"
TOOL = "tool"


# Graph Initialization.
graph_builder = StateGraph(ChatBotState)
graph_builder.add_node(CHATBOT, chatbot_node)
graph_builder.add_node(TOOL, ToolNode(tools = available_tools))
graph_builder.add_edge(TOOL, CHATBOT)
graph_builder.add_conditional_edges(
    CHATBOT,
    tool_call_required,
    path_map = {
        TOOL: TOOL,
        END: END
    }
)
graph_builder.set_entry_point(CHATBOT)
# Memory Initalization
memory = InMemorySaver()

# # If you want true persistence, you can use SqliteSaver instead.
# conn: sqlite3.Connection = sqlite3.connect("chatbot_memory.db")
# memory = SqliteSaver(connection = conn) # A sqlite db will be created if it does not exist. and store all the checkpoints there.


# Final Graph.
graph = graph_builder.compile(checkpointer = memory)

# display(Image(graph.get_graph().draw_mermaid_png()))



if __name__ == "__main__":

    config: Dict = {
        "configurable": {
            "thread_id": 1
        }
    }

    while True:
        human_input: str =  input("Human: ")
        if human_input in ["exit", "end"]:
            break

        response: ChatBotState = graph.invoke(
            ChatBotState(
                messages = [
                    HumanMessage(content = human_input)
                ]
            ),
            config = config 
        )

        print(f"AI: {response["messages"][-1].content}", end = "\n\n")




