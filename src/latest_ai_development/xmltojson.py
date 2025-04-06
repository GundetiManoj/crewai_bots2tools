import json
import xmltodict
import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool

# Load environment variables
load_dotenv()

# ------------------------------
# ✅ CrewAI Tool: XML to JSON Converter
# ------------------------------
@tool("xml_to_json_converter_tool")
def xml_to_json_converter_tool(xml_string: str) -> str:
    """Converts an XML string into JSON format."""
    try:
        # Convert XML to dictionary
        xml_dict = xmltodict.parse(xml_string)
        # Convert dictionary to JSON
        json_output = json.dumps(xml_dict, indent=4)
        return json_output
    except Exception as e:
        return f"Error: {e}"

# ------------------------------
# ✅ CrewAI Agent: XML Converter
# ------------------------------
xml_converter_agent = Agent(
    role="XML to JSON Converter",
    goal="Convert XML data into structured JSON format efficiently.",
    backstory=(
        "You are a highly skilled data transformer, specializing in converting XML data into JSON."
        " Your job is to ensure accuracy, structure, and error-free conversions."
    ),
    verbose=True,
    memory=True,
    tools=[xml_to_json_converter_tool]  # Attach the tool
)

# ------------------------------
# ✅ CrewAI Task: Convert XML to JSON
# ------------------------------
xml_conversion_task = Task(
    description="Convert the following XML data into JSON format: {xml_data}",
    expected_output="A properly formatted JSON output representing the XML data.",
    agent=xml_converter_agent
)

# ------------------------------
# ✅ CrewAI Execution: Run the Task
# ------------------------------
crew = Crew(
    agents=[xml_converter_agent],
    tasks=[xml_conversion_task],
    process=Process.sequential
)

# ------------------------------
# ✅ Run the Task with Example XML Input
# ------------------------------
if __name__ == "__main__":
    xml_input = """<library>
    <shelf>
        <book>
            <title>The Great Gatsby</title>
            <author>F. Scott Fitzgerald</author>
        </book>
        <book>
            <title>1984</title>
            <author>George Orwell</author>
        </book>
    </shelf>
    <shelf>
        <book>
            <title>To Kill a Mockingbird</title>
            <author>Harper Lee</author>
        </book>
    </shelf>
</library>"""
    inputs = {"xml_data": xml_input}
    result = crew.kickoff(inputs=inputs)
    
    print("\n🚀 Converted JSON Output:\n", result)
