import asyncio
from llama_index.core.agent.workflow import FunctionAgent

from common_utils import model, base_url, api_key

if "groq" in base_url:
    from llama_index.llms.groq import Groq as OpenAI
else:
    from llama_index.llms.openai import OpenAI


# Define a simple calculator tool
def multiply(a: float, b: float) -> float:
    """Useful for multiplying two numbers."""
    return a * b


# Create an agent workflow with our calculator tool
agent = FunctionAgent(
    llm=OpenAI(
        model=model,
        api_key=api_key,
        api_base=base_url,
        request_timeout=360.0,
        # Manually set the context window to limit memory usage
        context_window=8000,
    ),
    system_prompt="You are a helpful assistant that can use tools.",
    tools=[multiply],
)

async def main():
    # Run the agent
    response = await agent.run("What is 1234 * 4567?")
    print(str(response))


# Run the agent
if __name__ == "__main__":
    asyncio.run(main())
