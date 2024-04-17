from langchain_community.llms import Ollama
llm = Ollama(model="tinyllama")
query = "Tell me a joke"

for chunks in llm.stream(query):
    print(chunks, end='', flush=True)

print()

