import os
from dotenv import load_dotenv
from transformers import pipeline
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool

# Load environment variables
load_dotenv()

# ------------------------------
# ✅ CrewAI Tool: Document Comparison
# ------------------------------
@tool("document_comparison_tool")
def compare_documents(text1: str, text2: str) -> str:
    """
    Compares two documents and highlights differences.
    It identifies added, removed, and changed text.
    """

    # Load Hugging Face summarization model for text comparison
    summarizer = pipeline("text2text-generation", model="facebook/bart-large-cnn")

    # Create a structured prompt for comparison
    prompt = f"""
    Compare the following two documents and highlight the differences:
    
    Document 1:
    {text1}

    Document 2:
    {text2}

    Identify and describe the changes, including additions, deletions, and modifications.
    """

    # Generate comparison result
    result = summarizer(prompt, max_length=500, truncation=True)

    return result[0]['generated_text']

# ------------------------------
# ✅ CrewAI Agent: Document Comparator
# ------------------------------
comparison_agent = Agent(
    role="Document Comparison Expert",
    goal="Analyze and compare two documents to highlight differences.",
    backstory=(
        "An expert in text analysis, specializing in comparing different versions of documents "
        "to identify modifications and ensure accuracy."
    ),
    verbose=True,
    memory=True,
    tools=[compare_documents]  # Attach comparison tool
)

# ------------------------------
# ✅ CrewAI Task: Compare Two Documents
# ------------------------------
comparison_task = Task(
    description="Compare two documents:{text1} and {text2} and highlight all differences between them.",
    expected_output="A detailed report on differences between the two documents.",
    agent=comparison_agent
)

# ------------------------------
# ✅ CrewAI Execution: Running the Workflow
# ------------------------------
crew = Crew(
    agents=[comparison_agent],
    tasks=[comparison_task],
    process=Process.sequential
)

# ------------------------------
# ✅ Run the Task with Example Documents
# ------------------------------
if __name__ == "__main__":
    text1 = """This is the first version of a document.
    It has some content here.
    We are comparing it to another version."""

    text2 = """This is the second version of the document.
    Some content has been changed here.
    We are comparing it with a different version."""

    inputs = {"text1": text1, "text2": text2}
    result = crew.kickoff(inputs=inputs)

    print("\n🚀 Comparison Result:\n", result)
