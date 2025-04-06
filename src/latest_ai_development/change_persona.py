import os
import torch
import transformers
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool

# Define the Change Persona Tool
@tool("change_persona_tool")
def change_persona_tool(input_text: str, persona: str) -> str:
    """Rewrites the input text in the style of a specified persona."""
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
    You are an AI that transforms text into different writing styles based on a given persona.

    **Input Text:** {input_text}
    **Persona:** {persona}

    Rewrite the text as if it were written by the specified persona. Maintain coherence, structure, and style accordingly.

    **Output:**
    """

    # Generate rewritten text
    result = pipeline(prompt, max_length=len(input_text) + 100, num_return_sequences=1, temperature=0.7)

    return result[0]["generated_text"]


# Create an agent that utilizes the Change Persona Tool
persona_agent = Agent(
    role="Text Stylist",
    goal="Rewrite text in different personas while maintaining clarity and coherence.",
    backstory="A language expert capable of mimicking various writing styles effortlessly.",
    verbose=True,
    memory=True,
    tools=[change_persona_tool]
)

# Define a task where the agent will transform text into a given persona's style
persona_task = Task(
    description="Rewrite the {input_text} in the style of the specified {persona}.",
    expected_output="A well-structured rewritten text that fully embodies the given persona's writing style.",
    agent=persona_agent
)

# Assemble the CrewAI workflow
crew = Crew(
    agents=[persona_agent],
    tasks=[persona_task],
    process=Process.sequential
)

# Running the task with an example input
if __name__ == "__main__":
    inputs = {
        "input_text": "Write the methodology of solving dijkstra's algorithm.",
        "persona": "A student of 7th semester of IIT Bombay"
    }
    result = crew.kickoff(inputs=inputs)
    print("\nRewritten Text in Persona Style:\n")
    print(result)
