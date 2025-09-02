import os
from agent_config import AgentConfig, MemoryType
from beeai_framework.tools.search.wikipedia import WikipediaTool
from beeai_framework.tools.search.duckduckgo import DuckDuckGoSearchTool
# Import other tools as needed

class AgentPresets:
    """Predefined configurations for common agent types"""
    
    @staticmethod
    def researcher() -> AgentConfig:
        return AgentConfig(
            model_name=os.getenv("OPENAI_CHAT_MODEL", "gpt-3.5-turbo"),
            tools=[WikipediaTool(), DuckDuckGoSearchTool()],
            instructions="Find accurate information and provide factual data back.",
            role="A diligent researcher",
            env_vars={"OPENAI_API_KEY": ""},
            max_iterations=15
        )
    
    @staticmethod
    def analyst() -> AgentConfig:
        return AgentConfig(
            model_name=os.getenv("OPENAI_CHAT_MODEL", "gpt-4"),
            tools=[],  # Analysis might not need external tools
            instructions="Analyze data and provide insights with clear reasoning.",
            role="A analytical thinker",
            memory_type=MemoryType.TOKEN,
            memory_config={"max_tokens": 4000}
        )
    
    @staticmethod
    def recruiter() -> AgentConfig:
        return AgentConfig(
            model_name=os.getenv("OPENAI_CHAT_MODEL", "gpt-3.5-turbo"),
            tools=[],  # Add recruiting-specific tools
            instructions="Help identify requirements and qualifications for roles.",
            role="An experienced recruiter",
            env_vars={"OPENAI_API_KEY": ""}
        )