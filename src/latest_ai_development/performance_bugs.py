import os
import requests
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool

# Load API Key from .env
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Define the Code Bug Fixing Tool
@tool("code_perform_fixer_tool")
def code_perform_fixer_tool(code_snippet: str) -> str:
    """Analyzes a code snippet, identifies security issues, and provides a corrected version."""

    prompt = f"""
    You are a performance optimization expert. Analyze the given code snippet for inefficiencies and suggest an optimized version with explanations.

    *Input Code:*
    {code_snippet}

    *Performance Issues Found:*
    - [List all inefficiencies]

    *Optimized Version:*
    [Provide the improved, optimized code]

    *Explanation:*
    - [Explain performance improvements]
    """

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": "You are a tool to optimize the given code."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 2000,
        "temperature": 0.7
    }

    response = requests.post(GROQ_API_URL, headers=headers, json=payload)

    # Ensure correct response handling without JSON formatting
    if response.status_code == 200:
        return response.text.strip()  # No JSON parsing, just plain text output
    else:
        return f"Error: {response.status_code}, {response.text}"

# Create an agent that utilizes the Code Bug Fixing Tool
code_reviewer_agent = Agent(
    role="AI Code Reviewer",
    goal="Optimization of the given code.",
    backstory="A skilled software engineer specializing in debugging and optimizing the given code.",
    verbose=True,
    memory=True,
    tools=[code_perform_fixer_tool]
)

# Define a task where the agent will review and fix a given code snippet
code_review_task = Task(
    description="Analyze the given code:{code_snippet} snippet, identify the language and return the best optimized version of code.",
    expected_output="A optimized version of the code with a detailed explanation of the time complexities.",
    agent=code_reviewer_agent
)

# Assemble the CrewAI workflow
crew = Crew(
    agents=[code_reviewer_agent],
    tasks=[code_review_task],
    process=Process.sequential
)

# Running the task with an example input
if __name__ == "__main__":
    input_code = """
#include <iostream>
#include <vector>

// Function to check if there is a subset with the given sum
bool isSubsetSum(const std::vector<int>& set, int n, int targetSum) {
    // Base cases
    if (targetSum == 0) return true;  // A subset with sum 0 is always possible (empty subset)
    if (n == 0) return false;         // No elements left to process

    // If the last element is greater than the target sum, ignore it
    if (set[n - 1] > targetSum) {
        return isSubsetSum(set, n - 1, targetSum);
    }

    // Check if the target sum can be obtained by:
    // 1. Including the last element
    // 2. Excluding the last element
    return isSubsetSum(set, n - 1, targetSum) || 
           isSubsetSum(set, n - 1, targetSum - set[n - 1]);
}

int main() {
    std::vector<int> set = {3, 34, 4, 12, 5, 2};
    int targetSum = 9;
    int n = set.size();

    if (isSubsetSum(set, n, targetSum)) {
        std::cout << "Found a subset with the given sum" << std::endl;
    } else {
        std::cout << "No subset with the given sum" << std::endl;
    }

    return 0;
}
    """

    inputs = {"code_snippet": input_code}
    result = crew.kickoff(inputs=inputs)
    print("\nBug Fixing Results:\n")
    print(result)
