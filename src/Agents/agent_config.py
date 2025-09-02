from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Type
from beeai_framework.tools import AnyTool
from beeai_framework.memory import BaseMemory, UnconstrainedMemory
from beeai_framework.backend import ChatModel
from enum import Enum

class MemoryType(str, Enum):
    UNCONSTRAINED = "unconstrained"
    TOKEN = "token"
    SLIDING_WINDOW = "sliding_window"

class AgentConfig(BaseModel):
    # Core settings
    model_name: str = Field(default="gpt-3.5-turbo", description="LLM model to use")
    model_provider: str = Field(default="openai", description="Model provider (openai, anthropic, etc.)")
    provider_url: str = Field(default="", description="Custom provider URL if applicable")

    # Agent behavior
    role: Optional[str] = Field(default=None, description="Agent's role/persona")
    instructions: str = Field(default="", description="System instructions for the agent")
    
    # Tools and capabilities
    tools: List[AnyTool] = Field(default_factory=list, description="Tools available to the agent")
    
    # Memory configuration
    memory_type: MemoryType = Field(default=MemoryType.UNCONSTRAINED, description="Type of memory to use")
    memory_config: Dict[str, Any] = Field(default_factory=dict, description="Memory-specific configuration")
    
    # Template configuration
    input_schema: Optional[Type[BaseModel]] = Field(default=None, description="Input validation schema")
    prompt_template: Optional[str] = Field(default=None, description="Custom prompt template")
    
    # Environment and runtime
    env_vars: Dict[str, str] = Field(default_factory=dict, description="Required environment variables")
    max_iterations: int = Field(default=10, description="Maximum reasoning iterations")
    
    class Config:
        arbitrary_types_allowed = True

