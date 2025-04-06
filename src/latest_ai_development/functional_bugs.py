# import os
# import torch
# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
# from crewai import Agent, Task, Crew, Process
# from crewai.tools import tool
# import openai
# from dotenv import load_dotenv
# import requests
# # Load a lightweight Hugging Face model optimized for code tasks
# MODEL_NAME = "Salesforce/codet5-small"

# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)


# # Define the Code Bug Fixing Tool
# @tool("code_bug_fixer_tool")
# def code_bug_fixer_tool(code_snippet: str) -> str:
#     """Analyzes a code snippet, identifies functional issues, and provides a corrected version."""
    
#     # Construct the prompt
#     prompt = f"""
#     You are an AI code reviewer. Your task is to analyze the given code snippet, identify functional issues, and provide a corrected version with explanations.

#     *Input Code:*
#     {code_snippet}

#     *Issues Identified:*
#     - [List any functional bugs]

#     *Corrected Code:*
#     python
#     [Provide the corrected version]

#     *Explanation:*
#     - [Explain the fixes]
#     """

#     # Tokenize input and generate response
#     inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
#     output = model.generate(**inputs, max_length=1024)
#     generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

#     return generated_text
    
# # Create an agent that utilizes the Code Bug Fixing Tool
# code_reviewer_agent = Agent(
#     role="AI Code Reviewer",
#     goal="Find and fix functional bugs in Python code.",
#     backstory="A skilled software engineer specializing in debugging and fixing code issues.",
#     verbose=True,
#     memory=True,
#     tools=[code_bug_fixer_tool]
# )

# # Define a task where the agent will review and fix a given code snippet
# code_review_task = Task(
#     description="Analyze the given code:{code_snippet} snippet, identify functional issues, and return a corrected version.",
#     expected_output="A corrected version of the code with a detailed explanation of the fixes.",
#     agent=code_reviewer_agent
# )

# # Assemble the CrewAI workflow
# crew = Crew(
#     agents=[code_reviewer_agent],
#     tasks=[code_review_task],
#     process=Process.sequential
# )

# # Running the task with an example input
# if __name__ == "__main__":
#     input_code = """
#     #include <iostream>

# // Definition for singly-linked list.
# struct ListNode {
#     int val;
#     ListNode *next;
#     ListNode(int x) : val(x), next(nullptr) {}
# };

# class Solution {
# public:
#     ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
#         ListNode* dummyHead = new ListNode(0);
#         ListNode* current = dummyHead;
#         int carry = 0;

#         while (l1 != nullptr || l2 != nullptr) {
#             int x = (l1 != nullptr) ? l1->val : 0;
#             int y = (l2 != nullptr) ? l2->val : 0;
#             int sum = carry + x + y;
#             carry = sum / 10;
#             current->next = new ListNode(sum % 10);
#             current = current->next;

#             if (l1 != nullptr) l1 = l1->next;
#             if (l2 != nullptr) l2 = l2->next;
#         }

#         if (carry > 0) {
#             current->next = new ListNode(carry);
#         }

#         return dummyHead->next; // Bug: Memory leak, dummyHead is not deleted
#     }
# };

# int main() {
#     // Example usage
#     ListNode* l1 = new ListNode(2);
#     l1->next = new ListNode(4);
#     l1->next->next = new ListNode(3);

#     ListNode* l2 = new ListNode(5);
#     l2->next = new ListNode(6);
#     l2->next->next = new ListNode(4);

#     Solution solution;
#     ListNode* result = solution.addTwoNumbers(l1, l2);

#     // Print result
#     while (result != nullptr) {
#         std::cout << result->val << " ";
#         result = result->next;
#     }
#     std::cout << std::endl;

#     // Memory cleanup
#     // Bug: Memory for l1, l2, and result is not freed
#     return 0;
# }
#     """
#     inputs = {"code_snippet": input_code}
#     result = crew.kickoff(inputs=inputs)
#     print("\nBug Fixing Results:\n")
#     print(result)



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
@tool("code_bug_fixer_tool")
def code_bug_fixer_tool(code_snippet: str) -> str:
    """Analyzes a code snippet, identifies functional issues, and provides a corrected version."""

    prompt = f"""
    You are an AI code reviewer. Your task is to analyze the given code snippet, 
    identify functional issues, and provide a corrected version with explanations.

    **Input Code:**
    ```
    {code_snippet}
    ```

    **Issues Identified:**
    - List any functional bugs.

    **Corrected Code:**
    ```
    Provide the corrected version.
    ```

    **Explanation:**
    - Explain the fixes in detail.
    """

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": "You are a tool to find and fix functional bugs in given code."},
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
    goal="Find and fix functional bugs in code.",
    backstory="A skilled software engineer specializing in debugging and fixing code issues.",
    verbose=True,
    memory=True,
    tools=[code_bug_fixer_tool]
)

# Define a task where the agent will review and fix a given code snippet
code_review_task = Task(
    description="Analyze the given code:{code_snippet} snippet, identify functional issues, and return a corrected version.",
    expected_output="A corrected version of the code with a detailed explanation of the fixes.",
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
SELECT department_id, employee_id, salary
FROM (
    SELECT department_id, employee_id, salary,
           ROW_NUMBER() OVER (PARTITION BY department_id ORDER BY salary DESC) as rank
    FROM employees
) as ranked_employees
WHERE rank <= 3
ORDER BY department_id, salary DESC;
    """

    inputs = {"code_snippet": input_code}
    result = crew.kickoff(inputs=inputs)
    print("\nBug Fixing Results:\n")
    print(result)
