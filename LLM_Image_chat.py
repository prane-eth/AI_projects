# %% [markdown]
# #### Project: Chat with an image
# Simply upload an image and ask your question

# %% [markdown]
# #### Imports

# %%

from common_functions import ensure_llama_running, load_data_from_url, clean_prompt, convert_list_to_base64

import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama

ensure_llama_running()
load_dotenv()
llm_model = os.getenv('IMAGE_LLM_MODEL')


HOSTING_MODE = True

# %% [markdown]
# #### Streamlit workflow

# %%
image_paths = None
default_user_question = "What is in this image?"
default_link = 'https://raw.githubusercontent.com/mohammadimtiazz/standard-test-images-for-Image-Processing/master/standard_test_images/fruits.png'

if HOSTING_MODE:
	import streamlit as st
	st.set_page_config(
		page_title="Chat with an image",
		page_icon="🐍",
		# layout="centered",
		# initial_sidebar_state="expanded",
	)
	st.title('Chat with an image')
	st.header('Upload an image')
	st.write(f'[No image? Download one here]({default_link})')

	image_files = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
	if not image_files:
		st.stop()

	st.image(image_files)
	user_question = st.text_input('Ask a question about the image:', value=default_user_question)
	if not len(user_question):
		st.stop()

	image_paths = []
	for image_file in image_files:
		image_path = os.path.join('__pycache__', image_file.file_id + image_file.name)
		with open(image_path, 'wb') as file:
			file.write(image_file.getbuffer())
		image_paths.append(image_path)


# %% [markdown]
# #### Data collection

# %%
if not HOSTING_MODE:
	image_paths = [load_data_from_url(
		default_link,
		filename='test.jpg',
		return_path = True,
	)]
	user_question = default_user_question

image_b64s = convert_list_to_base64(image_paths)


# %% [markdown]
# #### Creating the model

# %%
llm = Ollama(model=llm_model)

# %% [markdown]
# Define the prompt

# %%
prompt = """
Image is attached by the user. Please answer the question user asks.
"""
prompt = clean_prompt(prompt)

# %%
promptTemplate = ChatPromptTemplate.from_messages([
	("system", prompt),
	("user", "{question}"),
])
llm_with_image_context = llm.bind(images=image_b64s)
chain = promptTemplate | llm_with_image_context | StrOutputParser()
result = chain.invoke({ "question": user_question })

if not result:
	result = "Sorry, no response"

if HOSTING_MODE:
	st.write(result)
else:
	print(result)

