# Created to Research on new AI agents that can be created, and added to the stack

from beeai_framework.agents.experimental import RequirementAgent
from beeai_framework.backend import ChatModel
from beeai_framework.memory import UnconstrainedMemory


# TODO: Create a more dynamic way to import models
model = ""

llm = ChatModel.from_name(model)
