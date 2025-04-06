import os
from dotenv import load_dotenv
from transformers import pipeline
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool

# Load environment variables
load_dotenv()

# ------------------------------
# ✅ CrewAI Tool: Document Classification
# ------------------------------
@tool("document_classification_tool")
def classify_document(text: str) -> str:
    """
    Classifies a document into predefined categories using a Hugging Face model.

    Categories:
    - News
    - Sports
    - Finance
    - Technology
    - Health
    - Entertainment
    """

    # Load the Hugging Face text classification model
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    # Define possible categories
    labels = ["News", "Sports", "Finance", "Technology", "Health", "Entertainment"]

    # Perform classification
    result = classifier(text, candidate_labels=labels)
    
    # Get the highest scoring category
    predicted_category = result["labels"][0]
    
    return f"The document belongs to the class: {predicted_category}"

# ------------------------------
# ✅ CrewAI Agent: Document Classifier
# ------------------------------
classifier_agent = Agent(
    role="Document Classification Specialist",
    goal="Analyze and classify documents into appropriate categories.",
    backstory=(
        "An expert in text classification, capable of understanding document context "
        "and categorizing it accurately into predefined categories."
    ),
    verbose=True,
    memory=True,
    tools=[classify_document]  # Attach classification tool
)

# ------------------------------
# ✅ CrewAI Task: Classify a Document
# ------------------------------
classification_task = Task(
    description="Analyze the given document:{text} and classify it into one of the predefined categories.",
    expected_output="A classification result indicating the category of the document.",
    agent=classifier_agent
)

# ------------------------------
# ✅ CrewAI Execution: Running the Workflow
# ------------------------------
crew = Crew(
    agents=[classifier_agent],
    tasks=[classification_task],
    process=Process.sequential
)

# ------------------------------
# ✅ Run the Task with Example Document
# ------------------------------
if __name__ == "__main__":
    document_text = """The stock market continues to fluctuate as investors await the Federal Reserve's next decision on interest rates.
    Experts predict that the economy will continue to face challenges, but opportunities still exist for long-term investors."""

    inputs = {"text": document_text}
    result = crew.kickoff(inputs=inputs)

    print("\n🚀 Classification Result:\n", result)
