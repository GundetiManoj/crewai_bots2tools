import os
import torch
import transformers
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool

# Define the elaboration tool
@tool("elaboration_tool")
def elaboration_tool(input_text: str, word_limit: int = 500) -> str:
    """Expands the given input text by adding relevant details and explanations to the specified length."""
    model_id = "EleutherAI/gpt-neo-125m"
    
    # Load the pipeline
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto"
    )
    
    prompt = f"""You are an elaborate assistant. 
    Your task is to expand the given input text to a specified length by adding relevant details, explanations, and context while staying on topic.
    Input: {input_text} in {word_limit} words
    Output:"""
    
    # Generate elaboration
    result = pipeline(prompt, max_length=word_limit + 250, num_return_sequences=1, temperature=0.7)
    
    return result[0]["generated_text"]


# Create an agent that utilizes the elaboration tool
elaboration_agent = Agent(
    role="Text Expander",
    goal="Generate detailed and contextually rich expansions for given topics.",
    backstory="A highly intelligent AI with expertise in expanding text while maintaining relevance and clarity.",
    verbose=True,
    memory=True,
    tools=[elaboration_tool]
)

# Define a task where the agent will elaborate on a given input
elaboration_task = Task(
    description="Expand the given {input_text} to a detailed and informative version within {word_limit} words.",
    expected_output="A well-expanded text with relevant details and proper context.",
    agent=elaboration_agent
)

# Assemble the CrewAI workflow
crew = Crew(
    agents=[elaboration_agent],
    tasks=[elaboration_task],
    process=Process.sequential
)

# Running the task with an example input
if __name__ == "__main__":
    inputs = {"input_text": "Tell me detailly about republic day", "word_limit": 500}
    result = crew.kickoff(inputs=inputs)
    print("\nGenerated Elaborated Text:\n")
    print(result)
