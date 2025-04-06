from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from latest_ai_development.tools.too import translation_tool, summarization_tool

@CrewBase
class LatestAiDevelopment:
    """LatestAiDevelopment crew"""

    @agent
    def researcher(self) -> Agent:
        return Agent(
            role='Researcher',
            goal='Conduct thorough research on the given topic',
            backstory='An expert researcher with extensive knowledge in gathering and analyzing information',
            tools=[
                translation_tool,
                summarization_tool
            ],
            verbose=True
        )

    @agent
    def reporting_analyst(self) -> Agent:
        return Agent(
            role='Reporting Analyst',
            goal='Create comprehensive reports from research findings',
            backstory='A skilled analyst who can transform raw information into structured, insightful reports',
            tools=[
                summarization_tool,
                translation_tool
            ],
            verbose=True
        )

    @task
    def research_task(self) -> Task:
        return Task(
            description="Conduct a thorough research about {topic}. "
                        "Use web search tools to gather information. "
                        "Use translation tool if needed to handle foreign language content. "
                        "Use summarization tool to condense key insights.",
            expected_output="A list with 10 bullet points of the most relevant information about {topic}. "
                            "Include translated and summarized insights if applicable.",
            agent=self.researcher,
            input_key='topic'
        )

    @task
    def reporting_task(self) -> Task:
        return Task(
            description="Review the research findings and create a comprehensive report. "
                        "Use summarization tool to refine key points. "
                        "Ensure the report is well-structured and informative.",
            expected_output="A detailed report with main topics, each expanded into a full section. "
                            "Formatted as a clear, readable markdown document.",
            agent=self.reporting_analyst
        )

    @crew
    def crew(self) -> Crew:
        """Creates the LatestAiDevelopment crew"""
        return Crew(
            agents=[self.researcher, self.reporting_analyst],
            tasks=[self.research_task, self.reporting_task],
            process=Process.sequential,
            verbose=True,
        )