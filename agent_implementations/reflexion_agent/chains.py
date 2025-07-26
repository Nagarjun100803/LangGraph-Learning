from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ibm import ChatWatsonx
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from config import settings
from models import ReviseAnswer, AnswerQuestion
from datetime import datetime
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from dotenv import load_dotenv


load_dotenv(dotenv_path = ".env", verbose = True) # To make sure the langsmith env credentials loaded. 

# LLM initailization.
# llm = ChatWatsonx(
#     apikey = settings.watsonx_apikey,
#     project_id = settings.watsonx_project_id,
#     url = settings.watsonx_url,
#     temperature = 0.9,
#     model_id = "ibm/granite-3-3-8b-instruct",
#     max_tokens = 10000
# )

llm = ChatGoogleGenerativeAI(
    google_api_key = settings.google_api_key,
    model = "gemini-2.0-flash",
    max_tokens = 10000,
    temperature = 0.9
)

# Actor prompt template.
# actor_prompt_template = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             "You are an expert AI Researcher. Current Time {current_time} "
#             "1, {initial_instruction} "
#             "2, Reflect and critique your answer. Be serve to maximize improvement. "
#             "3, After the reflection, **list 1-3 search queries separately** for researching improvements. Do not include them inside the reflection. "
#         ),
#         MessagesPlaceholder(variable_name = "messages"),
#         (
#             "system",
#             "Answer the user's question above using the required format. "
#             "**You Must always return valid JSON fenced by a markdown code block. Do not return any additional text. Also please check the response ended with a proper json termination.**"
#             # We instructing the llm to return the result in a JSON format, but not a actual json object. llm mostly return str object. But we need a
#             # proper json format even it is a str, that will help us to parse in PydanticToolsParser to get a str result as JSON/Pydantic scheme.
#         )

#     ]
# )

actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert AI Researcher. Current time: {current_time}.\n\n"
            "Follow these three steps:\n"
            "1. {initial_instruction}\n"
            "2. Reflect on your previous answer. Critique it clearly and suggest specific improvements.\n"
            "3. Generate **1–3 search queries** for refining the answer. ⚠️ List them **separately**, not inside the reflection.\n"
        ),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Now revise the answer using your reflection and output it as a JSON **string only**, with no markdown, no explanation, and no extra text.\n\n"
            "**Output must be a single valid JSON object starts and ends with curly braces.**\n"
            "**Do not wrap it in code blocks or include any commentary.**"
        )
    ]
)


# Initalizing the Responder Chain.
responder_chain = (
    actor_prompt_template.partial(current_time = datetime.now().isoformat(), initial_instruction = "Provide a details 250 words of answer. ") 
    | llm.bind_tools(tools = [AnswerQuestion], tool_choice = "AnswerQuestion")
)


# Initial Instruction for revisor chain.
# revisor_initial_instruction: str =  """
#     Revise your previous answer using the new information.
#         - You should use the previous critique to add important information to your answer.
#         - Your MUST include numerical citations in your revised answer to measure it can be verified.
#         - Add a "References" section to the bottom of your answer(which does not count towards the word limit). In the format of **https url with a citiation number**.
#             example - [1] https://example1.com
#                     - [2] https://example2.com
#         -You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 words.
# """

revisor_initial_instruction: str = """
You are revising your previous answer using the new information and critique provided.

Your revision MUST follow these guidelines:

1. Incorporate important missing information based on the critique.
2. Remove any superfluous or repetitive content.
3. Keep the revised answer concise — under 250 words (excluding references).
4. **You MUST include at least two numerical citations in the form [1], [2], etc., within the body of the answer.**
5. Add a "References" section at the end of the answer in the following strict format (required):

References:
[1] https://example1.com  
[2] https://example2.com  

Each citation must point to a valid HTTPS URL (beginning with "https://") and must correspond to the numbered citations used in the answer body.

⚠️ Do not invent references. If you don’t have real sources, create realistic placeholder HTTPS URLs in the correct format.

Ensure the references list and citations are clearly visible, verifiable, and match exactly.
"""


# Initalizing Revisor chain.

revisor_chain = (
    actor_prompt_template.partial(
        current_time = datetime.now().isoformat(),
        initial_instruction = revisor_initial_instruction
    )
    | llm.bind_tools(tools = [ReviseAnswer], tool_choice = "ReviseAnswer")
)   


# Validators to ensure the result is in proper pydantic schema.
response_chain_validator = PydanticToolsParser(tools = [AnswerQuestion])
revisor_chain_validator  = PydanticToolsParser(tools = [ReviseAnswer])


if __name__ == "__main__":

    "Testing this two chain response."

    messages: list[BaseMessage] =  []
    inital_human_message: HumanMessage = HumanMessage("Write a LinkedIn blog post in the topic of **LangGraph - A Game Changer in Agentic AI World**")

    messages.append(inital_human_message)
    
    response: AIMessage = responder_chain.invoke({"messages": messages})
    # Since we use llm.bind_tool() it store the results in a **.tool_calls** attribute not in a **.content**
    # Here at this stage we need to validate the llm response. whether it give us a output in a 
    # proper Pydantic[AnswerQuestion] scheme. So use **response_chain_validator**. It internally uses PydanticToolsParser.

    # We will do the same for revisor_chain output.

    answer: list[AnswerQuestion] = response_chain_validator.invoke(response)

    json_like_answer: str = answer[0].model_dump_json(indent = 4)
    

    print(f"The AI response from Responder Chain: {json_like_answer}", end = "\n\n\n")
    # change the 
    ai_response = AIMessage(content = json_like_answer)

    messages.append(ai_response)
    print(f"AI message added to messages. messages list contains {len(messages)} BaseMessage", end = "\n\n\n")


    revisor_response: AIMessage = revisor_chain.invoke({"messages": messages})

    revised_answer: list[ReviseAnswer] = revisor_chain_validator.invoke(revisor_response)

    json_like_revised_answer: str = revised_answer[0].model_dump_json(indent = 4)

    print(f"The final AI response from Revisor Chain, \n{json_like_revised_answer}")






    

