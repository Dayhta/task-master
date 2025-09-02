import os 
from typing import Dict, Type, Optional
from beeai_framework.angets.react import ReActAgent
from beeai_framework.agents.experimental import RequirementAgent
from beeai_framwork.backend import ChatModel
from beeai_framework.memory import UnconstrainedMemory, TokenMemory
from beeai_framework.memory import AnyTool
from agent_config import AgentConfig, MemoryType
from dotenv import load_dotenv

load_dotenv()

class AgentFactory:
    """
    Factory for creating different types of agents based on imported configuration
    """

    AGENT_TYPES = {
        "ReAct": ReActAgent,
        "Requirement": RequirementAgent
    }

    MEMORY_TYPES = {
        MemoryType.UNCONSTRAINED: UnconstrainedMemory,
        MemoryType.TOKEN: TokenMemory
    }

    @classmethod
    def create_memory(cls, config: AgentConfig):
        """Creates memory instance based on imported config"""
        memory_class = cls.MEMORY_TYPES.get(config.memory_type, UnconstrainedMemory)
        return memory_class(**config.memory_config)

    @classmethod
    def create_llm(cls, config: AgentConfig):
        """Creates the LLM Instance based on imported config with provider URL support"""
        model_name = config.model_name or os.getenv("OPENAI_CHAT_MODEL", "gpt-3.5-turbo")
        
        # Handle different providers
        if config.provider_url:
            # Use custom provider URL (e.g., OpenRouter)
            return ChatModel.from_name(
                model_name,
                base_url=config.provider_url,
                api_key=cls._get_api_key_for_provider(config.model_provider)
            )
        elif config.model_provider == "openrouter":
            # OpenRouter specific configuration
            return ChatModel.from_name(
                model_name,
                base_url="https://openrouter.ai/api/v1",
                api_key=os.getenv("OPENROUTER_API_KEY")
            )
        elif config.model_provider == "anthropic":
            # Anthropic specific configuration
            return ChatModel.from_name(
                model_name,
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )
        else:
            # Default to OpenAI
            return ChatModel.from_name(
                model_name,
                api_key=os.getenv("OPENAI_API_KEY")
            )

    @classmethod
    def _get_api_key_for_provider(cls, provider: str) -> str:
        """Get API key based on provider"""
        provider_key_map = {
            "openai": "OPENAI_API_KEY",
            "openrouter": "OPENROUTER_API_KEY", 
            "anthropic": "ANTHROPIC_API_KEY",
            "custom": "CUSTOM_API_KEY"
        }
        
        env_var = provider_key_map.get(provider.lower(), "OPENAI_API_KEY")
        api_key = os.getenv(env_var)
        
        if not api_key:
            raise ValueError(f"Missing API key for provider '{provider}'. Set environment variable: {env_var}")
        
        return api_key


    @classmethod
    def validate_env_vars(cls, config: AgentConfig):
        """Validates environment variables based on imported config"""
        missing_vars = []
        
        # Check config-specific env vars
        for var_name, default_value in config.env_vars.items():
            if not os.getenv(var_name, default_value):
                missing_vars.append(var_name)
        
        # Check provider-specific API key
        try:
            cls._get_api_key_for_provider(config.model_provider)
        except ValueError as e:
            missing_vars.append(str(e))
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {missing_vars}")
        
    @classmethod
    async def create_agent(cls, agent_type: str = "ReAct", config: Optional[AgentConfig] = None) -> ReActAgent:
        """Creates an agent instance based on the agent type and configuration"""
        if agent_type not in cls.AGENT_TYPES:
            raise ValueError(f"Unknown agent type: {agent_type}. Available types: {list(cls.AGENT_TYPES.keys())}")
        
        if config is None:
            config = AgentConfig()

        cls.validate_env_vars(config)

        memory = cls.create_memory(config)
        llm = cls.create_llm(config)

        agent_class = cls.AGENT_TYPES.get(agent_type, ReActAgent)
        agent_kwargs = {
            "llm": llm,
            "memory": memory,
            "tools": config.tools,
        }

        if config.role:
            agent_kwargs["role"] = config.role
        if config.instructions:
            agent_kwargs["instructions"] = config.instructions
        if hasattr(config, 'max_iterations'):
            agent_kwargs["max_iterations"] = config.max_iterations
        return agent_class(**agent_kwargs)