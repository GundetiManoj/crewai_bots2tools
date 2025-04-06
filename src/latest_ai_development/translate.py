import asyncio
from crewai.tools import BaseTool
from crewai import Agent, Task 
from pydantic import BaseModel, Field
from googletrans import Translator
class TranslationBotInput(BaseModel):
    text: str = Field(..., description="Text to translate")
    target_language: str = Field(..., description="Target language code")


# --- TranslationBot ---
class TranslationBot(BaseTool):
    name: str = "TranslationBot"
    description: str = "Translate text into the desired language."
    args_schema: type = TranslationBotInput

    async def _run(self, text: str, target_language: str) -> str:
        """
        Asynchronous translation method
        """
        try:
            # Initialize the translator
            translator = Translator()

            # Directly await the translation coroutine
            translation = await translator.translate(text, dest=target_language)

            # Return the translated text
            return translation.text
        except Exception as e:
            return f"Translation failed: {str(e)}"


# --- TranslationTask ---
class TranslationTask(Task):
    tool_name: str = "TranslationBot"
    inputs_schema: type = TranslationBotInput
    description: str = "A task to translate text from one language to another."
    expected_output: str = "Translated text in the target language."

    async def execute(self, input_data):
        # Prepare the inputs for the translation tool
        text = input_data["text"]
        target_language = input_data["target_language"]

        # Initialize the TranslationTool
        translation_tool = TranslationBot()

        # Run the translation asynchronously
        translated_text = await translation_tool._run(text, target_language)
        return translated_text


# --- Workflow with Agent ---
class TranslationAgent(Agent):
    name: str = "TranslationAgent"
    role: str = "Translate text from one language to another"
    goal: str = "Perform accurate and fast translations"
    backstory: str = "The agent is responsible for translating text using an online translation service"

    tasks: list = [TranslationTask]

    async def run(self, input_data):
        task = TranslationTask()
        translated_text = await task.execute(input_data)
        return translated_text


# --- Example Usage ---
async def main():
    # Example text and language to translate
    input_data = {
        "text": "Hello, how are you ?",
        "target_language": "te"
    }

    # Initialize the agent (which will use the translation tool)
    agent = TranslationAgent()

    # Run the agent to perform the translation
    translated_text = await agent.run(input_data)

    # Print the result
    print("\nTranslated Text:", translated_text)


# Run the async function
if __name__ == "__main__":
    asyncio.run(main())