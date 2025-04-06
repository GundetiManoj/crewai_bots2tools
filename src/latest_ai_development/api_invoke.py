import os
import requests
import litellm
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
import json
# Load environment variables
load_dotenv()

# Set up LiteLLM with API key
LITELLM_API_KEY = os.getenv("GROQ_API_KEY")  # Ensure your .env file has this variable set

# ------------------------------
# ✅ CrewAI Tool: API Invoke Tool (With LLM Validation)
# ------------------------------
@tool("api_invoke_tool")
def api_invoke_tool(url: str, method: str = "GET", headers: dict = None, body: dict = None) -> str:
    """Validates and invokes an API request using an LLM for error checking."""

    headers = headers or {}  # Ensure headers are not None
    body = body or {}  # Ensure body is not None

    # Construct validation prompt
    validation_prompt = f"""
    You are an API request validator. Given the API details:

    **URL:** {url}
    **Method:** {method}
    **Headers:** {headers}
    **Body:** {body}

    Check for errors and return a corrected version if needed.
    """

    # Validate API request details using LiteLLM
    validated_input = litellm.completion(
        model="groq/llama-3.3-70b-versatile",  # Using a more efficient Llama model
        messages=[{"role": "system", "content": "You validate API request inputs before execution."},
                  {"role": "user", "content": validation_prompt}],
        api_key=LITELLM_API_KEY,
    )["choices"][0]["message"]["content"].strip()
    try:
            validated_data = json.loads(validated_input)  # Convert JSON string to dict
            url = validated_data.get("url", url)  # Use corrected values if provided
            method = validated_data.get("method", method).upper()
            headers = validated_data.get("headers", headers)
            body = validated_data.get("body", body)
    except json.JSONDecodeError:
            return f"Error: Failed to parse validated input from LLM. Response:\n{validated_input}"
    # Make API request after validation
    try:
        if method.upper() == "GET":
            response = requests.get(url, headers=headers)
        elif method.upper() == "POST":
            response = requests.post(url, json=body, headers=headers)
        elif method.upper() == "PUT":
            response = requests.put(url, json=body, headers=headers)
        elif method.upper() == "DELETE":
            response = requests.delete(url, headers=headers)
        else:
            return "Error: Unsupported HTTP method."

        return response.text  # Return API response as text
    except Exception as e:
        return f"Error: {str(e)}"

# ------------------------------
# ✅ CrewAI Agent: API Request Handler
# ------------------------------
api_invoker_agent = Agent(
    role="API Validator & Executor",
    goal="Validate and execute API requests accurately.",
    backstory=(
        "You are an API specialist who ensures that all requests are properly formatted "
        "before execution. Your expertise helps prevent errors and optimize API calls."
    ),
    verbose=True,
    memory=True,
    tools=[api_invoke_tool]  # Attach the tool
)

# ------------------------------
# ✅ CrewAI Task: API Invocation
# ------------------------------
api_invocation_task = Task(
    description="Validate and invoke an API request to {api_url} using method {http_method}.",
    expected_output="A validated API response.",
    agent=api_invoker_agent
)

# ------------------------------
# ✅ CrewAI Execution: Running the Workflow
# ------------------------------
crew = Crew(
    agents=[api_invoker_agent],
    tasks=[api_invocation_task],
    process=Process.sequential
)

# ------------------------------
# ✅ Run the Task with Example API Call
# ------------------------------
if __name__ == "__main__":
    api_url = "https://jsonplaceholder.typicode.com/posts"
    http_method = "GET"

    inputs = {"api_url": api_url, "http_method": http_method}
    result = crew.kickoff(inputs=inputs)

    print("\n🚀 API Response:\n", result)
