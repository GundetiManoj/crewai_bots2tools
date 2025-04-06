import os
import torch
import transformers
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool

# Define the Review-Highlight Tool
@tool("review_highlight_tool")
def review_highlight_tool(input_text: str) -> str:
    """Analyzes the input text for spelling mistakes, grammar issues, and factual errors."""
    model_id = "EleutherAI/gpt-neo-125m"

    # Load the text generation pipeline
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto"
    )

    # Construct the prompt
    prompt = f"""
    You are an expert proofreader and fact-checker. Review the following text and highlight:
    1. Spelling mistakes with corrections
    2. Grammar issues with corrections
    3. Factual errors (with correct information if possible)

    **Input Text:**
    {input_text}

    **Output:**
    - List of spelling mistakes with corrections
    - Grammar issues with corrections
    - Factual errors with correct information
    """

    # Generate review report
    result = pipeline(prompt, max_length=len(input_text) + 200, num_return_sequences=1, temperature=0.5)

    return result[0]["generated_text"]


# Create an agent that utilizes the Review-Highlight Tool
proofreader_agent = Agent(
    role="Proofreader & Fact-Checker",
    goal="Analyze text for errors and provide corrections.",
    backstory="An experienced linguist and researcher, dedicated to ensuring clarity, accuracy, and correctness in writing.",
    verbose=True,
    memory=True,
    tools=[review_highlight_tool]
)

# Define a task where the agent will review a given text
review_task = Task(
    description="Analyze the given input:{input_text} for spelling, grammar, and factual errors, and provide corrections.",
    expected_output="A detailed review highlighting errors and their corrections.",
    agent=proofreader_agent
)

# Assemble the CrewAI workflow
crew = Crew(
    agents=[proofreader_agent],
    tasks=[review_task],
    process=Process.sequential
)

# Running the task with an example input
if __name__ == "__main__":
    input_text = " The sun rotates arond the earth and it's distence is 93 milions miles."
    inputs = {"input_text": input_text}
    result = crew.kickoff(inputs=inputs)
    print("\nReview Results:\n")
    print(result)
