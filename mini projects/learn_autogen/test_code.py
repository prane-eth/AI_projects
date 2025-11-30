# pip install -U "autogen-agentchat" "autogen-ext[openai]"
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelFamily

from common_utils import model

async def main() -> None:
    agent = AssistantAgent("assistant", OpenAIChatCompletionClient(model=model,
							model_info={
								"vision": False,
								"function_calling": False,
								"json_output": False,
								"family": ModelFamily.GPT_5,
								"structured_output": True,
							}))
    print("Agent initialized.")
    response = await agent.run(task="Say 'Hello World!'")
    print("Agent response:", response.messages[-1].content)  # type: ignore

asyncio.run(main())
