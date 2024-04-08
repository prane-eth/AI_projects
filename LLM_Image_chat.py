# %% [markdown]
# #### Project: Chat with an image

# %% [markdown]
# #### Imports

# %%

from common_functions import ensure_llama_running, load_data_from_url

import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from llama_index.readers.file import ImageReader

import streamlit as st

ensure_llama_running()
load_dotenv()
llm_model = os.getenv('IMAGE_LLM_MODEL')

# %% [markdown]
# #### Streamlit workflow

# %%
st.title('Chat with an image')

st.header('Upload an image')

st.write('[No image? Download here](https://raw.githubusercontent.com/mohammadimtiazz/standard-test-images-for-Image-Processing/master/standard_test_images/HappyFish.jpg)')

file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])

if not file:
	st.stop()

st.image(file, use_column_width=True)

user_question = st.text_input('Ask a question about the image:')

if not len(user_question):
	st.stop()


# # %% [markdown]
# # #### Data collection

# # %%
# image_path = load_data_from_url(
#     "https://raw.githubusercontent.com/mohammadimtiazz/standard-test-images-for-Image-Processing/master/standard_test_images/HappyFish.jpg",
#     return_path = True,
# )
# reader = ImageReader()
# image = reader.load_data(image_path)

# image

# # %% [markdown]
# # #### Creating the model

# # %%
# llm = Ollama(model=llm_model)
# llm

# # %% [markdown]
# # Define the prompt

# # %%
# # prompt can also be saved to a file and used as a template
# prompt = """
# Image is attached below. Please answer the question user asks.
# """

# prompt = prompt.strip().replace("\n", " ")

# # %%
# promptTemplate = ChatPromptTemplate.from_messages([
# 	("system", prompt),
# 	("user", "{image}"),
# 	("user", "{question}"),
# ])
# chain = promptTemplate | llm | StrOutputParser()  # Parse output as string
# # Different operations are chained together to form a 'pipeline'.
# # The output of one operation is passed as input to the next.

# result = chain.invoke({ "question": "What is in the image?.", "image": image })

# display(Markdown(f"**Response:**\n\n{result}"))

