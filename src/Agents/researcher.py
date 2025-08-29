from beeai_framework.agents.react import ReActAgent
from beeai_framework.backend import ChatModel
from beeai_framework.memory import UnconstrainedMemory
from beeai_framwork.tools.search.wikipedia import WikipediaTool


# TODO: Create a more dynamic way to import models
model = ""

llm = ChatModel.from_name(model)

Researcher = ReActAgent(
    llm=llm,
    memory=UnconstrainedMemory(),
    tools=[WikipediaTool()],
    role="A diligent researcher",
    instructions="Find accurate information and then provide that data back"
)