import os

from groq import Groq

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Tell me a joke",
        }
    ],
    model="llama2-70b-4096",
)

print(chat_completion.choices[0].message.content)
