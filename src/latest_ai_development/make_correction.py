import os
import torch
import transformers
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool

# Define the Correction Tool
@tool("correction_tool")
def correction_tool(input_text: str) -> str:
    """Corrects the input text by fixing spelling, grammar, and factual errors."""
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
    You are an AI editor. Your task is to correct the given text by fixing:
    - Spelling mistakes
    - Grammar issues
    - Factual errors (if any)

    Ensure that the corrected text maintains proper structure, clarity, and meaning.

    **Input Text:**
    {input_text}

    **Corrected Output:**
    """

    # Generate corrected text
    result = pipeline(prompt, max_length=len(input_text) + 200, num_return_sequences=1, temperature=0.5)

    # Extract the corrected text from the result
    corrected_text = result[0]["generated_text"].split("**Corrected Output:**")[-1].strip()

    return corrected_text

# Create an agent that utilizes the Correction Tool
correction_agent = Agent(
    role="Text Corrector",
    goal="Correct text for spelling, grammar, and factual errors.",
    backstory="An AI editor with expertise in ensuring text clarity, accuracy, and correctness.",
    verbose=True,
    memory=True,
    tools=[correction_tool]
)

# Define a task where the agent will correct a given text
correction_task = Task(
    description="Correct the given input: {input_text} for spelling, grammar, and factual errors.",
    expected_output="A corrected version of the input text with all errors fixed.",
    agent=correction_agent
)

# Assemble the CrewAI workflow
crew = Crew(
    agents=[correction_agent],
    tasks=[correction_task],
    process=Process.sequential
)

# Running the task with an example input
if __name__ == "__main__":
    input_text = "The sun rotates arond the earth and it's distence is 93 milions miles."
    inputs = {"input_text": input_text}
    result = crew.kickoff(inputs=inputs)
    print("\nCorrected Text:\n")
    print(result)