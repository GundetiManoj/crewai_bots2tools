from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import asyncio
from googletrans import Translator
import requests
import os

# --- TranslationBot ---
class TranslationBotInput(BaseModel):
    text: str = Field(..., description="Text to translate")
    target_language: str = Field(..., description="Target language code")


class TranslationBot(BaseTool):
    name: str = "TranslationBot"
    description: str = "Translate text into the desired language."
    args_schema: Type[BaseModel] = TranslationBotInput

    async def _run(self, text: str, target_language: str) -> str:
        try:
            translator = Translator()
            # Use asyncio.to_thread to run the synchronous translate method in a separate thread
            translation = await asyncio.to_thread(translator.translate, text, target_language)
            return translation.text
        except Exception as e:
            return f"Translation failed: {str(e)}"


# --- SummarizationBot ---
class SummarizationBotInput(BaseModel):
    text: str = Field(..., description="Text to summarize")
    length: str = Field("medium", description="Summary length: short, medium, or long")


class SummarizationBot(BaseTool):
    name: str = "SummarizationBot"
    description: str = "Summarize text using Groq API."
    args_schema: Type[BaseModel] = SummarizationBotInput

    async def _run(self, text: str, length: str = "medium") -> str:
        try:
            # Groq API key from environment
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                return "Groq API key is missing."

            # Define the Groq API endpoint (you may need to update the endpoint)
            endpoint = "https://api.groq.com/v1/completions"  # Verify with Groq's API documentation

            # Set max tokens for summary length
            max_tokens = 100 if length == "short" else 200 if length == "medium" else 400

            payload = {
                "model": "groq-llama-2-7b",  # Example model, adjust if needed
                "prompt": f"Summarize the following text: {text}",
                "max_tokens": max_tokens,
                "temperature": 0.7
            }

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }

            # Make the request to Groq API
            response = requests.post(endpoint, json=payload, headers=headers)

            if response.status_code == 200:
                return response.json()["choices"][0]["text"].strip()
            else:
                return f"Failed to summarize: {response.status_code} - {response.text}"
        except Exception as e:
            return f"Summarization failed: {str(e)}"


# --- MyCustomTool --- (Updated for both Translation and Summarization)
class MyCustomToolInput(BaseModel):
    """Input schema for MyCustomTool."""
    argument: str = Field(..., description="Description of the argument.")


class MyCustomTool(BaseTool):
    name: str = "MyCustomTool"
    description: str = (
        "Clear description for what this tool is useful for, your agent will need this information to use it."
    )
    args_schema: Type[BaseModel] = MyCustomToolInput

    async def _run(self, argument: str) -> str:
        """
        Implementation of the custom tool.
        This is where you define what your tool does.
        """
        # Example logic for using TranslationBot or SummarizationBot inside MyCustomTool.
        if argument == "translate":
            translation_input = {
                "text": "Hello, how are you?",
                "target_language": "es"  # Example: Spanish
            }
            translation_bot = TranslationBot()
            translated_text = await translation_bot._run(**translation_input)
            return f"Translation Result: {translated_text}"
        
        elif argument == "summarize":
            summarization_input = {
                "text": "This is an example text to summarize for your custom tool. It can be a long paragraph.",
                "length": "short"
            }
            summarization_bot = SummarizationBot()
            summary = await summarization_bot._run(**summarization_input)
            return f"Summary: {summary}"

        return "No valid operation found for the given argument."


# Example of how the tool can be used
async def test_custom_tool():
    tool = MyCustomTool()

    # Example of calling the tool with "translate"
    result = await tool._run(argument="translate")
    print(result)  # Output the translated result

    # Example of calling the tool with "summarize"
    result = await tool._run(argument="summarize")
    print(result)  # Output the summarized result


if __name__ == "__main__":
    asyncio.run(test_custom_tool())
