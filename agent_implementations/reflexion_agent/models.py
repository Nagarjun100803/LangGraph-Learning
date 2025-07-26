from pydantic import BaseModel, Field

class Reflection(BaseModel):
    "Represents the Reflection"
    missing: str = Field(description = "Critique of what is missing.")
    superfluous: str = Field(description = "Critique of what is superfluous.")


class AnswerQuestion(BaseModel):
    "Represents the Answer"
    answer: str = Field(description = "250 words of detailed answer to this question.")
    search_queries: list[str] = Field(
        description = "1-3 search queries for researching improvements" \
        " to address the citique of your current answer."
    )
    reflection: Reflection = Field(description = "Your reflection to the initial answer.")


class ReviseAnswer(AnswerQuestion):
    "Represents the ReviseAnswer schema."
    citations: list[str] = Field(
        description = "Citations motivating your updated answer."
    ) 