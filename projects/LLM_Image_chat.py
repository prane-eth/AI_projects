# $ streamlit run LLM_Image_chat.py 

# %% [markdown]
# #### Project: Chat with an image
# Simply upload an image and ask your question

# %% [markdown]
# #### Imports

# %%

from common_functions import get_ollama, load_data_from_url, \
			clean_prompt, convert_list_to_base64, supported_image_formats

import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
llm_model = os.getenv('LLM_VISION_MODEL')

# Demo: ../Demo/LLM_Image_chat.mp4 and ../Demo/LLM_Image_chat2.mp4

HOSTING_MODE = True

# %% [markdown]
# #### Streamlit workflow

# %%
image_files = None
default_user_question = "What are in this image?"
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

	image_files = st.file_uploader('Upload an image', type=supported_image_formats, accept_multiple_files=True)
	if not image_files:
		st.stop()

	st.image(image_files)
	user_question = st.text_input('Ask a question about the image:', value=default_user_question)
	if not len(user_question):
		st.stop()


# %% [markdown]
# #### Data collection

# %%
if not HOSTING_MODE:
	image_files = [load_data_from_url(
		default_link,
		filename='test.jpg',
		return_path = True,
	)]
	user_question = default_user_question

image_b64s = convert_list_to_base64(image_files)


# %% [markdown]
# #### Creating the model

# %%
llm = get_ollama(llm_model)

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
chain = promptTemplate | llm.bind(images=image_b64s) | StrOutputParser()
result = chain.invoke({ "question": user_question })

if not result:
	result = "Sorry, no response"

if HOSTING_MODE:
	st.write(result)
else:
	print(result)

