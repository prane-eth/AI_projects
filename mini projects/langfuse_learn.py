from langfuse import observe
from langfuse.openai import openai  # type: ignore

from common_utils import model

@observe()
def story():
    return openai.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "What is Langfuse?"}],
    ).choices[0].message.content

@observe()
def main():
    return story()

if __name__ == "__main__":
	response = main()
	print("Response:", response)
	print("Run completed and observed in Langfuse.")
