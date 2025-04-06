import os
import sys
import warnings
from datetime import datetime
from crewai import Agent, Crew, Process, Task
from crewai.tools import tool
from crewai_tools import SerperDevTool
# import litellm

# Suppress warnings
warnings.filterwarnings("ignore")

# API Keys
os.environ["GROQ_API_KEY"] = "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"
os.environ["SERPER_API_KEY"] = "90e9267b3f1c50a570ee9686807885a0ddcfdb52"
os.environ["OPENAI_API_KEY"] = "sk-dummy-key-to-bypass-validation"

# Configure LiteLLM with multiple fallback providers
# litellm.set_verbose = False
# litellm.fallback_providers = ["openai", "azure", "anthropic"]
# litellm.success_callback = []
# litellm.failure_callback = []

# Custom Tools
@tool("Advanced Web Research Tool")
def advanced_web_research_tool(query: str) -> str:
    """
    Perform comprehensive web research using Serper and fallback methods
    
    Args:
        query (str): Search query
    
    Returns:
        str: Research findings
    """
    try:
        # Use SerperDevTool for primary research
        serper_tool = SerperDevTool()
        serper_results = serper_tool.search(query)
        
        # If Serper fails, provide fallback research
        if not serper_results:
            fallback_sources = [
                f"Wikipedia summary of '{query}'",
                f"Recent academic papers about {query}",
                f"Industry reports related to {query}"
            ]
            return "\n".join([
                "Fallback Research Findings:",
                f"1. {fallback_sources[0]}",
                f"2. {fallback_sources[1]}",
                f"3. {fallback_sources[2]}"
            ])
        
        # Process and summarize Serper results
        processed_results = []
        for i, result in enumerate(serper_results[:5], 1):
            processed_results.append(f"{i}. {result.get('title', 'Untitled')}: {result.get('snippet', 'No description')}")
        
        return "\n".join([
            "Web Research Findings:",
            *processed_results
        ])
    except Exception as e:
        return f"Research failed: {str(e)}"

@tool("Summarization Tool")
def summarization_tool(text: str, max_words: int = 100) -> str:
    """
    Summarize text by reducing to specified word count
    
    Args:
        text (str): Text to summarize
        max_words (int, optional): Maximum words. Defaults to 100.
    
    Returns:
        str: Summarized text
    """
    try:
        words = text.split()
        summary = " ".join(words[:max_words]) + "..." if len(words) > max_words else text
        return summary
    except Exception as e:
        return f"Summarization failed: {str(e)}"

class LatestAiDevelopment:
    # def __init__(self):
        # Flexible LLM configuration with fallbacks
        # self.llm_config = {
        #     "model": "groq/llama-3.3-70b-versatile",
        #     "temperature": 0.7,
        #     "max_tokens": 4096,
        #     "fallback_models": [
        #         "gpt-3.5-turbo",
        #         "claude-2.1"
        #     ]
        # }

    def create_researcher_agent(self):
        return Agent(
            role='AI Research Specialist',
            goal='Conduct comprehensive research on emerging AI technologies',
            backstory='An expert researcher with deep knowledge of AI trends and innovations',
            tools=[
                SerperDevTool(),  # Primary web search tool
                advanced_web_research_tool  # Advanced research tool with fallback
            ],
            # llm_config=self.llm_config,
            verbose=True,
            allow_delegation=True
        )

    # def create_reporting_agent(self):
    #     return Agent(
    #         role='AI Insights Analyst',
    #         goal='Transform research findings into structured, insightful reports',
    #         backstory='A skilled analyst who can distill complex information into clear, actionable insights',
    #         tools=[
    #             summarization_tool
    #         ],
    #         llm_config=self.llm_config,
    #         verbose=True,
    #         allow_delegation=True
    #     )

    def create_research_task(self, agent, topic):
        return Task(
            description=f"Conduct an in-depth research on {topic}, "
                        "utilizing web search and advanced research tools. "
                        "Gather comprehensive information from multiple sources.",
            expected_output=f"A detailed report on {topic} with at least 10 key insights, "
                            "including web search results and summarized information.",
            agent=agent,
            input_key='topic'
        )

    # def create_reporting_task(self, agent):
    #     return Task(
    #         description="Review and synthesize the research findings into a comprehensive report. "
    #                     "Use summarization tool to refine key points and ensure clarity.",
    #         expected_output="A well-structured markdown report that provides deep insights "
    #                         "and actionable intelligence on the research topic.",
    #         agent=agent
    #     )

    def run_crew(self, topic):
        # Create agents
        researcher = self.create_researcher_agent()
        # reporting_analyst = self.create_reporting_agent()

        # Create tasks
        research_task = self.create_research_task(researcher, topic)
        # reporting_task = self.create_reporting_task(reporting_analyst)

        # Create and run crew
        crew = Crew(
            agents=[researcher],#[researcher, reporting_analyst],
            tasks=[research_task],#[research_task, reporting_task]
            process=Process.sequential,
            verbose=True,
            memory=True
        )

        # Execute the crew
        return crew.kickoff(inputs={'topic': topic})

def main():
    # Topic of research
    research_topic = 'AI LLMs in 2024: Latest Developments and Future Trends'

    try:
        # Initialize and run the crew
        ai_development_crew = LatestAiDevelopment()
        result = ai_development_crew.run_crew(research_topic)

        # Print and save the result
        print("\n--- Research Insights ---")
        print(result)

        # Save to file
        with open('ai_llms_research.md', 'w', encoding='utf-8') as f:
            f.write(result)
        print("\nResearch saved to ai_llms_research.md")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()