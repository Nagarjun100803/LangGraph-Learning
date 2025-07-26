from typing import Any
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage
from langchain_tavily import TavilySearch
from config import settings
import json



tavily_search = TavilySearch(
    max_results = 2,
    tavily_api_key = settings.tavily_api_key
)

def execute_tool(state: list[BaseMessage]) -> list[BaseMessage]:

    # Get the last AIMessage from the state.
    last_ai_message: AIMessage = state[-1]

    if (not hasattr(last_ai_message, "tool_calls")) or (not last_ai_message.tool_calls):
        # if the last LLM call did not use any tools we append empty list.
        return state + []
    
    tool_messages: list[ToolMessage] = []

    for tool_call in last_ai_message.tool_calls:
        tool_call_id: str = tool_call["id"] # Id of the tool call.
        # Get the search queries from tool call args.
        search_queries: list[str] = tool_call["args"].get("search_queries", []) 

        # Search for search_queries using Tavily.
        query_results: dict[str, dict[str, Any]] =  {}

        for query in search_queries:
            
            query_result: dict = tavily_search.invoke(query)
            query_results[query] = query_result

        tool_messages.append(
            ToolMessage(
                content = json.dumps(query_results),
                tool_call_id = tool_call_id
            )
        )
    
    return tool_messages



    
if __name__ == "__main__":

    # Testing the execute tool method with sample state.

    test_state: list[BaseMessage] = [

    HumanMessage(content = "Write a linkedin blog in the topic of **LangGraph the Game Changer**"),
    AIMessage(
            content = "",
            tool_calls = [
                {
                    'name': 'AnswerQuestion',
                    'args': {
                        'answer': "## Title: LangGraph - Revolutionizing Language Learning as a Game Changer ### Introduction In the ever-evolving landscape of technology, innovation often stems from creatively blending disparate fields to tackle age-old challenges. One such groundbreaking convergence is found in LangGraph, a novel game-based platform designed to transform language acquisition into an engaging and efficient experience. This blog post explores LangGraph's unique features, its impact on language learning, and why it stands out as a game changer in education.\n\n### Unconventional Approach - Combining Gaming with Linguistics\nLangGraph distinguishes itself by integrating the compelling elements of video games with the structured demands of language learning. Traditional methods often face challenges with maintaining learner motivation and engagement. LangGraph addresses these issues through an interactive schema that rewards progress and encourages continuous learning.\n\n#### Features of LangGraph\n1. **Adaptive Learning Paths:** LangGraph tailors learning modules to individual users' needs, adapting to their proficiency levels and interests, ensuring optimized learning.\n2. **Gamified Progression:** Users advance through levels, unlock content, and earn points by completing exercises, integrating the satisfaction of achievement derived from gaming.\n3. **Social Interaction:** A built-in community enables peer-to-peer learning and friendly competition, motivating users to practice and refine their language skills in a supportive environment.\n4. **Multilingual Support:** LangGraph caters to a diverse user base by supporting multiple languages, breaking down barriers and fostering global linguistic fluency.\n\n### A Scientific Leap in Cognitive Development and Memory Retention\nResearch indicates that gamification significantly enhances cognitive functions, particularly memory retention and critical thinking. LangGraph's mechanisms, such as spaced repetition, flashcards, and interactive puzzles, align with these findings, facilitating deeper language absorption and recall.\n\n### Transforming Education and Career Prospects\nLanguage skills are a pivotal asset in an increasingly interconnected world. LangGraph not only democratizes access to quality language education but also equips learners with essential 21st-century competencies, enhancing employability across sectors.\n\n### Conclusion - LangGraph: The Future of Language Acquisition\nLangGraph stands out as a pioneering platform that synergistically merges gaming and linguistics, offering a dynamic, adaptive, and fruitful learning experience. Its commitment to engaging users through game mechanics while harnessing the power of technology positions it uniquely as a game changer in the realm of language education.\n\n### Reflection\n**Missing:** While the blog outlines LangGraph's benefits, it lacks concrete examples of its real-world impact and user testimonials to bolster credibility.\n**Superfluous:** The introduction could be more engaging by starting with a captivating anecdote illustrating the challenges of traditional language learning.\n\n### Search Queries\n1. 'LangGraph case studies' \n2. 'User reviews of LangGraph as a language learning tool'\n3. 'LangGraph success stories'",
                        'reflection': {
                            'missing': "The blog lacks real-world examples, case studies, and direct user testimonials which could substantially enhance its credibility and demonstrate LangGraph's tangible impact on learners' lives.",
                            'superfluous': "The current introduction doesn't immediately grab attention with a compelling narrative or hook. Tweaking the opening to include an engaging personal anecdote or a bold statement about language learning challenges could make the post more captivating from the start."
                        },
                        'search_queries': [
                            'LangGraph case studies',
                            'User reviews of LangGraph as a language learning tool',
                            'LangGraph success stories'
                        ]
                    },
                    'id': 'chatcmpl-tool-d20b54da35fa40d598bd167c77dac474',
                    'type': 'tool_call'
                }
            ]
        )
    ]

    response = execute_tool(test_state)

    print(f"The state contains {len(response)} BaseMessage", end = "\n\n")

    print(f"The tool message:\n\t{response[-1].to_json()}\n\n\n{response[-1]}")


