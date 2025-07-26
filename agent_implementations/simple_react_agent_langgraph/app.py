from fastapi import FastAPI
from graph import graph, AgentState


def get_answer(input: str) -> dict:
    response: AgentState = graph.invoke(
        AgentState(
            input = input,
            agent_outcome = None,
            intermediate_steps = []
        )
    )
    return response["agent_outcome"].return_values

app = FastAPI(
    title = "ReAct Agent",
    summary = "A simple ReAct agent implementation using LangGraph",
    description = "This API provides a simple ReAct agent that can perform actions based on user input.",
    version = "0.1.0"
)


@app.get("/react_agent")
async def react_agent(input: str) -> dict:
    """
    Endpoint to get the answer from the ReAct agent.
    
    Args:
        input (str): The input string provided to the agent.
    
    Returns:
        dict: The agent's response containing the outcome.
    """
    return get_answer(input)
    


if __name__ == "__main__":

    import uvicorn
    uvicorn.run(app, port = 8000)