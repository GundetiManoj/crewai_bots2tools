from crewai import Agent, Task, Crew, Process
from form_parser_crew import DocumentAnalysisTool
from src.latest_ai_development.summarize import SummarizationTool
from crewai import LLM
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# # Initialize LLM (Ollama with LLaMA3)
# llm = LLM(
#     model="ollama/llama3:latest",
#     base_url="http://localhost:11434"
# )

# Initialize document parsing tool
my_tool = DocumentAnalysisTool(
    project_id="vision-454216",
    location="us",
    processor_id="e8d7e54d9f2335a3",
    gemini_api_key=os.getenv("GEMINI_API_KEY"),
    creds_path="vision-454216-e49450484a5a.json"
)

# Initialize summarization tool
summarization_tool = SummarizationTool()

# Define the agent
extraction_agent = Agent(
    role="Document Extraction Specialist",
    goal="Extract structured data and generate a short summary from PDF documents.",
    backstory="An AI expert specializing in document parsing and summarization using state-of-the-art models.",
    tools=[my_tool, summarization_tool],
    memory=True,
    verbose=True
    # llm=llm
)

# Define the task using the {file_path} input
extraction_task = Task(
    description="Extract structured data and a short summary from the PDF file at: {file_path}",
    expected_output="A JSON with fields like invoice number, vendor, total amount, and a 3-line summary.",
    agent=extraction_agent
)

# Setup the crew
crew = Crew(
    agents=[extraction_agent],
    tasks=[extraction_task],
    # manager_llm=llm,
    process=Process.sequential,
    verbose=True
)

# Run the crew with a PDF file input
if __name__ == "__main__":
    example_invoice_path = "BANK.pdf"  # Replace with your real file path
    inputs = {"file_path": example_invoice_path}
    result = crew.kickoff(inputs=inputs)

    print("\n📄 Final Extracted Output:\n")
    print(result)
