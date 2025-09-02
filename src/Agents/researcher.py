import asyncio
import logging
import os
import sys
from pydantic import BaseModel, Field
from beeai_framework.agents.react import ReActAgent
from beeai_framework.backend import ChatModel
from beeai_framework.memory import UnconstrainedMemory
from beeai_framwork.tools.search.wikipedia import WikipediaTool
from beeai_framework.tools.search.duckduckgo import DuckDuckGoSearchTool
from beeai_framework.tools import AnyTool
from agent_config import AgentConfig
from agent_factory import AgentFactory
from presets import AgentPresets
from dotenv import load_dotenv

# Created to Research on new AI agents that can be created, and added to the stack


load_dotenv()
# TODO: Create a more dynamic way to import models
class ResearchTaskInput(BaseModel):
    """Defines the input for the researcher agent."""
    topic: str = Field(description="The high-level topic to research")
    specifics: list[str] = Field(description="Specific points or questions to investigate")


# OpenRouter-based researcher configuration
researcher_config = AgentConfig(
    model_name=os.getenv("OPENROUTER_CHAT_MODEL", "openai/gpt-3.5-turbo"),
    model_provider="openrouter",
    provider_url="https://openrouter.ai/api/v1",
    tools=[WikipediaTool(), DuckDuckGoSearchTool()],
    instructions="Find accurate information and then provide that data back. Focus on factual, verifiable information from reliable sources.",
    role="A diligent researcher powered by OpenRouter",
    input_schema=ResearchTaskInput,
    env_vars={"OPENROUTER_API_KEY": ""},
    max_iterations=15
)


async def create_researcher_agent():
    """Create a researcher agent with OpenRouter configuration"""
    return await AgentFactory.create_agent("ReAct", researcher_config)

async def create_researcher_with_preset():
    """Create a researcher agent using preset configuration"""
    return await AgentFactory.create_agent("ReAct", AgentPresets.researcher_openrouter())

async def create_custom_researcher(provider_url: str, model_name: str):
    """Create a researcher with custom provider"""
    custom_config = AgentConfig(
        model_name=model_name,
        model_provider="custom", 
        provider_url=provider_url,
        tools=[WikipediaTool(), DuckDuckGoSearchTool()],
        instructions="Research with custom AI provider",
        role="Custom researcher",
        env_vars={"CUSTOM_API_KEY": ""}
    )
    return await AgentFactory.create_agent("ReAct", custom_config)
