from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
from googletrans import Translator
import requests
import os

class TranslationToolInput(BaseModel):
    text: str = Field(..., description="Text to translate")
    target_language: str = Field(default="en", description="Target language code")

class SummarizationToolInput(BaseModel):
    text: str = Field(..., description="Text to summarize")
    max_length: int = Field(default=100, description="Maximum length of summary")

class TranslationTool(BaseTool):
    name: str = "Translation Tool"
    description: str = "Translates text from one language to another"
    args_schema: Type[BaseModel] = TranslationToolInput

    def _run(self, text: str, target_language: str = 'en') -> str:
        """
        Translate text to target language
        """
        try:
            translator = Translator()
            translation = translator.translate(text, dest=target_language)
            return translation.text
        except Exception as e:
            return f"Translation failed: {str(e)}"

class SummarizationTool(BaseTool):
    name: str = "Summarization Tool"
    description: str = "Summarizes text by reducing its length"
    args_schema: Type[BaseModel] = SummarizationToolInput

    def _run(self, text: str, max_length: int = 100) -> str:
        """
        Summarize text by truncating to max_length words
        """
        try:
            words = text.split()
            summary = " ".join(words[:max_length]) + "..." if len(words) > max_length else text
            return summary
        except Exception as e:
            return f"Summarization failed: {str(e)}"

# Create tool instances
translation_tool = TranslationTool()
summarization_tool = SummarizationTool()