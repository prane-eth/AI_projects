from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory

import streamlit as st


st.title("Celebrity Search Results")
topic_name = st.text_input("Search the topic u want")
if topic_name:
	query = st.text_input("Enter the query on the topic")
if not topic_name or not query:
	import sys
	sys.exit(0)

# Prompt Templates

query_input_prompt = PromptTemplate(
	template="Answer the query on {topic_name}. Be very very short in 5-10 words. Query: {query}",
	input_variables=["topic_name", "query"],
)

# Memory

topic_memory = ConversationBufferMemory(input_key="topic_name", memory_key="chat_history")

llm = Ollama()
chain = LLMChain(
	llm=llm,
	prompt=query_input_prompt,
	verbose=True,
	output_key="query_answer",
	memory=topic_memory,
)

parent_chain = SequentialChain(
	chains=[chain],
	input_variables=["topic_name", "query"],
	output_variables=["query_answer"],
	verbose=True,
)

print('\n\n Called parent chain \n\n -------------------')
print(topic_name)

if topic_name:
    answer = parent_chain.invoke({"topic_name": topic_name, "query": query})
    st.write(answer['query_answer'])

    with st.expander("Topic name"):
        st.info(topic_memory.buffer)
else:
    print('\n\n Empty input_text \n\n -------------------')
