from langchain.agents import create_react_agent
from langchain.schema.runnable import Runnable
from typing import List
from langchain_core.tools import BaseTool
from config import settings
from tools import available_tools
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain import hub
from langchain_core.prompts import PromptTemplate
from langchain_core.agents import AgentAction, AgentFinish
from dotenv import load_dotenv
from langchain_core.exceptions import OutputParserException

load_dotenv() # To make sure all the env variables to Trace using LangSmith.


tools: List[BaseTool] = available_tools


llm = ChatGoogleGenerativeAI(
    google_api_key =  settings.google_api_key,
    model = "gemini-2.5-flash",
    max_tokens = 3000,
    temperature = 0.9
)


react_prompt_template: PromptTemplate = hub.pull("hwchase17/react")

agent: Runnable = create_react_agent(
    llm = llm,
    prompt = react_prompt_template,
    tools = tools
)


if __name__ == "__main__":

    sample_questions: List[str] = [
        "Hello, This is Nagarjun",
        "What is 2 plus 98",
        "Why langgraph is best in AI Frameworks."
    ]

    for sample_question in sample_questions:
        try:
            agent_outcome: (AgentAction | AgentFinish) = agent.invoke(
                {
                    "input": sample_question,
                    "intermediate_steps": [],
                    "agent_outcome": None
                }
            )  

            print(f"The agent outcome is {agent_outcome}", end = "\n\n")

        except OutputParserException as e:
            print("Cannot Parse the LLM result into AgentAction or AgentFinish.")
            print(f"LLM Result: {e.llm_output}", end = "\n\n")


        