from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import os
import requests

class SummarizationTool(BaseTool):  
    # Define required fields
    name: str = "SummarizationTool"  
    description: str = "A tool to summarize text using a large language model."
    api_key: str = Field(default_factory=lambda: os.getenv("GROQ_API_KEY"))
    model: str = "llama-3.3-70b-versatile"
    base_url: str = "https://api.groq.com/openai/v1/chat/completions"
    
    def _run(self, text: str, max_words: int = 50) -> str:
        """
        Summarize the given text using the summarization API.
        
        This method implements the abstract '_run' method.
        
        Args:
            text (str): Text to be summarized
            max_words (int, optional): Maximum number of words in summary
        
        Returns:
            str: Summarized text
        """
        try:
            # Validate input
            if not text or not isinstance(text, str):
                return "Error: Invalid input. Please provide a non-empty string."
            
            # Construct prompt for summarization
            prompt = f"""
            Summarize the following text concisely, capturing the most important points:
            
            TEXT:
            {text}
            
            Guidelines:
            - Preserve the main ideas and key insights
            - Be extremely concise
            - Limit the summary to approximately {max_words} words
            - Focus on the core message of the text
            """
            
            # Prepare API request payload
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system", 
                        "content": "You are a helpful assistant skilled in creating concise summaries."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                "temperature": 0.3,
                "max_tokens": 150  # Adjust based on max_words
            }
            
            # Make API call
            response = requests.post(
                self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json=payload
            )
            
            # Check response
            response.raise_for_status()
            
            # Extract summary
            result = response.json()
            summary = result['choices'][0]['message']['content'].strip()
            
            return summary
        
        except requests.RequestException as e:
            return f"API Request Error: {str(e)}"
        except Exception as e:
            return f"Summarization error: {str(e)}"
        

from crewai import Agent, Task
class SummarizationAgent:
    def __init__(self, summarization_tool: SummarizationTool):
        """
        Create a summarization agent that utilizes the SummarizationTool
        """
        self.summarization_tool = summarization_tool
        self.agent = Agent(
            role='Text Summarizer',
            goal='Summarize provided text using LLM',
            backstory='An expert at summarizing text with the help of LLMs',
            tools=[self.summarization_tool],
            verbose=True
        )
    
    def create_task(self, text: str, max_words: int):
        """
        Create a task to summarize the text
        """
        try:
            summary = self.summarization_tool._run(text, max_words)
            if not summary:
                raise ValueError("Summary could not be generated.")
        except Exception as e:
            return f"Error generating summary: {str(e)}"

        task = Task(
            description="Summarize the provided text",
            agent=self.agent,
            tools=[self.summarization_tool],
            expected_output=summary  # Add expected_output here
        )
        
        return summary  # Return the summary for output


def main():
    # Ensure your API key is in the environment or pass it directly
    summarization_tool = SummarizationTool(api_key=os.getenv("GROQ_API_KEY"))
    
    summarization_agent = SummarizationAgent(summarization_tool=summarization_tool)
    
    text = """
        Artificial Intelligence (AI) is transforming multiple industries like health care, finance, and transportation.
        It enables machines to learn from data, make decisions, and perform tasks that typically require human intelligence.
        The rapid advancements in AI technologies, such as deep learning and natural language processing, are driving innovation and efficiency.
        However, ethical considerations and potential job displacement are significant challenges that need to be addressed.
        Policymakers and industry leaders must collaborate to ensure that AI is developed and deployed responsibly.
        By harnessing the power of AI, we can unlock new opportunities for innovation, improve efficiency, and create a brighter future for humanity.
    """
    
    summarized_text = summarization_agent.create_task(text, max_words=50)
    print(summarized_text)

if __name__ == "__main__":
    main()
