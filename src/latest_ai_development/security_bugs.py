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
@tool("code_secure_fixer_tool")
def code_secure_fixer_tool(code_snippet: str) -> str:
    """Analyzes a code snippet, identifies security issues, and provides a corrected version."""

    prompt = f"""
    You are a security expert. Analyze the given code snippet for security vulnerabilities and provide a secure version with explanations.

    *Input Code:*
    {code_snippet}

    *Security Issues Found:*
    - [List all vulnerabilities]

    *Secure Version:*
    [Provide the corrected secure code]


    *Explanation:*
    - [Explain security fixes]
    """

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": "You are a tool to find and fix security vulnerabilities in given code."},
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
    goal="Find and fix security issues in code.",
    backstory="A skilled software engineer specializing in debugging and fixing privacy and security issues.",
    verbose=True,
    memory=True,
    tools=[code_secure_fixer_tool]
)

# Define a task where the agent will review and fix a given code snippet
code_review_task = Task(
    description="Analyze the given code:{code_snippet} snippet, identify security issues, and return a corrected version.",
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
from flask import Flask, request, render_template_string
import sqlite3

app = Flask(__name__)

# Vulnerable database connection
def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # SQL Injection Vulnerability
        conn = get_db_connection()
        query = f"SELECT * FROM users WHERE username = '{username}' AND password = '{password}'"
        user = conn.execute(query).fetchone()
        conn.close()
        
        if user:
            return "Login successful!"
        else:
            return "Invalid credentials!"
    
    # XSS Vulnerability
    return '''
        <form method="post">
            Username: <input type="text" name="username"><br>
            Password: <input type="password" name="password"><br>
            <input type="submit" value="Login">
        </form>
    '''

@app.route('/greet')
def greet():
    name = request.args.get('name', 'Guest')
    
    # XSS Vulnerability
    return render_template_string(f"<h1>Hello, {name}!</h1>")

if __name__ == '__main__':
    app.run(debug=True)
    """

    inputs = {"code_snippet": input_code}
    result = crew.kickoff(inputs=inputs)
    print("\nBug Fixing Results:\n")
    print(result)
