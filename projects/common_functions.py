import subprocess
import sys

def ensure_installed(required_packages):
	missing_packages = []
	for package_import_name in required_packages:
		pip_name = None
		if isinstance(package_import_name, dict):
			pip_name = list(package_import_name.keys())[0]
			package_import_name = package_import_name[pip_name]
		try:
			__import__(package_import_name)
		except ModuleNotFoundError:
			missing_packages.append(pip_name or package_import_name.replace('_', '-'))
		except Exception as e:
			print(f'Error importing {package_import_name}: {e}')
	if missing_packages:
		subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing_packages)


import os
import re
import shutil
from urllib.parse import urlparse

import zipfile
import wget
from environments_utils import is_notebook
import pandas as pd

import base64
import io
from PIL import Image

from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_experimental.tabular_synthetic_data.base import SyntheticDataGenerator
from langchain_experimental.tabular_synthetic_data.prompts import \
			SYNTHETIC_FEW_SHOT_PREFIX, SYNTHETIC_FEW_SHOT_SUFFIX



RANDOM_STATE = 42


current_dir = os.path.dirname(os.path.realpath(__file__))

datasets_dir = os.path.join(current_dir, 'datasets')
# Create the dataset folder if it doesn't exist
if not os.path.exists(datasets_dir):
	os.makedirs(datasets_dir)

cache_dir = os.path.join(current_dir, '__pycache__')
if not os.path.exists(cache_dir):
	os.makedirs(cache_dir)


def download_and_extract_dataset_zip(dataset_zip_url, required_files):
	# Check if the required files exist
	if all(os.path.exists(file) for file in required_files):
		return  # All required files exist.

	# Extract the dataset name from the URL
	dataset_zip_filename = os.path.basename(dataset_zip_url)

	# Download the dataset file if it doesn't exist
	dataset_zip_filepath = os.path.join(datasets_dir, dataset_zip_filename)
	if not os.path.exists(dataset_zip_filepath):
		# Download the dataset file
		print(f'Downloading {dataset_zip_filename}... ', end='', flush=True)
		wget.download(dataset_zip_url, out=dataset_zip_filepath)
		print('Downloaded.')

	# Extract the dataset file
	print(f'Extracting {dataset_zip_filename}... ', end='', flush=True)
	dataset_url_filepath = os.path.basename(dataset_zip_url)
	if dataset_url_filepath.endswith('.zip'):
		with zipfile.ZipFile(dataset_zip_filepath, 'r') as zip_ref:
			zip_ref.extractall(datasets_dir)
	elif dataset_url_filepath.endswith('.tar.gz'):
		shutil.unpack_archive(
			dataset_zip_filepath, extract_dir=datasets_dir, format='gztar'
		)
	elif dataset_url_filepath.endswith('.tar.bz2'):
		shutil.unpack_archive(
			dataset_zip_filepath, extract_dir=datasets_dir, format='bztar'
		)
	else:
		print(f'Unsupported file format: {dataset_url_filepath}')
		return
	print('Extracted. \n')

	# Remove the dataset zip file
	os.remove(dataset_zip_filepath)



def load_data_from_url(dataset_url, filename=None, return_path=False):
	if not filename:
		parsed_url = urlparse(dataset_url)
		filename = parsed_url.path.split('/')[-1]
	filepath = os.path.join(current_dir, datasets_dir, filename)
	if not os.path.exists(filepath):
		print(f'Downloading {filename}... ', end='', flush=True)
		wget.download(dataset_url, out=filepath)
		print('Downloaded.')
	if return_path:
		return filepath
	
	if filename.endswith('.csv') or filename.endswith('.tsv'):
		# imports should be only if used, without CPU wastage
		import pandas as pd

	if filename.endswith('.csv'):
		df = pd.read_csv(filepath)
		return df
	elif filename.endswith('.tsv'):
		df = pd.read_csv(filepath, sep='\t')
		return df

	# if other formats
	# raise ValueError(f'Unsupported file format: {filename}')
	try:
		with open(filepath, 'r') as file:
			text = file.read()
		return text
	except Exception as e:
		print(f'Error reading {filename}: {e}')
		raise ValueError(f'Error reading {filename}: {e}')


image_save_path = os.path.join(current_dir, 'images')

def save_plot(filename, plt, savingEnabled=True):
	if not savingEnabled:
		return
	if '.' not in filename:  # if no format, add .png
		filename += '.png'
	os.makedirs(image_save_path, exist_ok=True)
	plt.savefig(os.path.join(image_save_path, filename))


def evaluate_model(model, X_train, y_train, X_test, y_test, X, y):
	# imports should be only if used, without CPU wastage, only when relevant project is run
	import numpy as np
	np.random.seed(RANDOM_STATE)

	from sklearn.metrics import confusion_matrix, f1_score  # , roc_auc_score, average_precision_score
	# from sklearn.model_selection import cross_val_score

	# Evaluates a given model on training and testing data
	try:
		model.fit(X_train, y_train)

		y_test_pred = model.predict(X_test)

		# Evaluating on training set and testing set
		train_accuracy = model.score(X_train, y_train)
		test_accuracy = model.score(X_test, y_test)
		overfitting = train_accuracy - test_accuracy
		if overfitting < 0:  # Overfitting is positive or zero
			overfitting = 0

		# # Cross-validation scores
		# try: # Some models need y in string type
		# 	cv_scores = cross_val_score(model, X.values, y.astype('str'), cv=5, scoring='accuracy')
		# except ValueError:  # Some models don't accept string type and raise ValueError above
		# 	cv_scores = cross_val_score(model, X.values, y, cv=5, scoring='accuracy')
		# cv_accuracy = np.mean(cv_scores)

		# Accuracy from confusion matrix
		conf_matrix = confusion_matrix(y_test, y_test_pred)
		conf_matrix_accuracy = conf_matrix.diagonal().sum() / conf_matrix.sum()
		f1 = f1_score(y_test, y_test_pred, average='binary')

		# # ROC AUC Score
		# if hasattr(model, 'predict_proba'):
		# 	y_test_proba = model.predict_proba(X_test)[:, 1]
		# 	roc_auc = roc_auc_score(y_test, y_test_proba)
		# 	average_precision = average_precision_score(y_test, y_test_proba)
		# else:
		# 	roc_auc = None
		# 	average_precision = None

		return {
			'Test Accuracy': test_accuracy,
			'Train Accuracy': train_accuracy,
			'Overfitting value': overfitting,
			# 'CV Accuracy': cv_accuracy,
			'Confusion Matrix Accuracy': conf_matrix_accuracy,
			'F1-Score': f1,
			# 'ROC AUC Score': roc_auc,
			# 'Average Precision Score': average_precision,
		}
	except Exception as e:
		print(f'Error with model {model}: {e}')
		return None


def get_model_scores(models_to_try, X_train, y_train, X_test, y_test, X, y):
	import pandas as pd
	# Dictionary to store scores of the tested models
	model_scores = {}

	for name, model in models_to_try.items():
		result = evaluate_model(model, X_train, y_train, X_test, y_test, X, y)
		if result:
			model_scores[name] = result

	# Converting scores to DataFrame
	scores_df = pd.DataFrame.from_dict(model_scores, orient='index')
	scores_df.sort_values(by=['Test Accuracy', 'Train Accuracy'], ascending=False, inplace=True)  # , 'CV Accuracy'
	scores_df.reset_index(inplace=True)
	scores_df.rename(columns={'index': 'Model'}, inplace=True)
	return scores_df


def hyperparam_tuning(model, X_train, y_train, param_grid=None, silent=False):
	# imports should be only if used, without CPU wastage
	from sklearn.model_selection import GridSearchCV
	from sklearn.linear_model import LogisticRegression
	from sklearn.ensemble import GradientBoostingClassifier
	if param_grid is None:
		param_grid = {
			'iterations': [100, 200, 300],
			'learning_rate': [0.01, 0.05, 0.1],
			'depth': [4, 6, 8],
		}
	if isinstance(model, GradientBoostingClassifier):
		# rename or remove unsupported values
		if 'depth' in param_grid:
			param_grid['max_depth'] = param_grid.pop('depth')
		if 'iterations' in param_grid:
			param_grid['n_estimators'] = param_grid.pop('iterations')
	if isinstance(model, LogisticRegression):
		if 'iterations' in param_grid:
			param_grid['max_iter'] = param_grid.pop('iterations')
		if 'learning_rate' in param_grid:
			param_grid.pop('learning_rate')
		if 'depth' in param_grid:
			param_grid.pop('depth')
	param_grid['random_state'] = [RANDOM_STATE]
	
	# Define and fit the grid search
	scoring_metric = 'accuracy'
	grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring=scoring_metric, verbose=1)
	grid_search.fit(X_train, y_train)

	# Best parameters and best score
	if not silent:
		print('Best Parameters:', grid_search.best_params_)
		print('Best', scoring_metric, 'Score:', grid_search.best_score_)

	# Return the best estimator
	return grid_search.best_estimator_


def host_chainlit(notebook_file, HOSTING_MODE=True):
	if not HOSTING_MODE:
		print('Hosting mode is set to False')
		return
	if not is_notebook():
		# Allow only in notebook mode
		# If running directly as python file, no need to create a new python file
		return

	# copy .env and common_functions from current_dir to cache_dir
	shutil.copy(os.path.join(current_dir, '.env'), cache_dir)
	shutil.copy(os.path.join(current_dir, 'common_functions.py'), cache_dir)

	script_file = os.path.join(cache_dir, notebook_file)
	# delete file before creating a new one
	if os.path.exists(script_file):
		os.remove(script_file)
	os.system(f'jupyter nbconvert {notebook_file} --to script --output {script_file}')
	script_file += '.py'
	try:
		
		# os.system(f'chainlit run {output_file}')
		subprocess.run(['chainlit', 'run', script_file], check=True)
	except KeyboardInterrupt:
		print("Interrupted by user")


def ensure_ollama_running():
	import requests
	try:
		requests.get('http://localhost:11434/')
	except:
		print('LLAMA is not running. Please start LLAMA first.')
		raise Exception('LLAMA is not running. Please start LLAMA first.')


def get_notebook_name(vscode_path, default_filename):
	
	if vscode_path:
		value = os.path.basename(vscode_path)
		if value:
			return value

	ensure_installed(['ipyparams', 'ipynbname', 'ipynb_path'])
	try:
		import ipyparams

		value = ipyparams.notebook_name
		if value:
			return ipyparams.notebook_name
	except:
		pass

	try:
		import ipynbname

		value = ipynbname.name()
		if value:
			return value
	except:
		pass

	try:
		import ipynb_path

		value = ipynb_path.get()
		if value:
			return os.path.basename(value)
	except:
		pass

	try:
		import ipynb_path

		value = ipynb_path.get(__name__)
		if value:
			return os.path.basename(value)
	except:
		pass

	return default_filename


def clean_prompt(prompt, llm=None):
	# remove comments and clean up the prompt to reduce tokens
	prompt = re.sub(r'#.*', '', prompt)  # remove comments

	if llm:
		# Print number of tokens in the prompt
		print('Number of tokens before cleanup:', llm.get_num_tokens(prompt))

	prompt = re.sub(r'\n+', '\n', prompt)  # remove extra newlines where there are more than one
	prompt = '\n'.join([line.strip() for line in prompt.split('\n')])  # strip each line
	prompt = prompt.strip()
	# remove punctuations at the start and end of the prompt
	punctuations = ',.!?'
	while prompt[0] in punctuations:
		prompt = prompt[1:]
	while prompt[-1] in punctuations:
		prompt = prompt[:-1]
	prompt = prompt.replace('\'s', 's')  # replace 's with s to save token usage for '
	for article in ['a', 'an', 'the', 'please']:  # remove 'a ', 'an ', 'the '
		prompt = prompt.replace(article + ' ', '')
		prompt = prompt.replace(article.capitalize() + ' ', '')

	if llm:
		print('Number of tokens after cleanup:', llm.get_num_tokens(prompt))

	return prompt


def display_md(text):
	from IPython.display import display, Markdown
	display(Markdown(text))


def shorten_prompt(input_prompt):
	# imports should be only if used, without CPU wastage
	from dotenv import load_dotenv
	from langchain_core.prompts import ChatPromptTemplate
	from langchain_core.output_parsers import StrOutputParser
	from langchain_community.llms import Ollama

	ensure_ollama_running()
	load_dotenv()
	llm_model = os.getenv('LLM_MODEL')
	llm = Ollama(model=llm_model)
	input_token_count = llm.get_num_tokens(input_prompt)
	print('Initial number of tokens:', input_token_count)

	shortener_prompt = '''
		You are a Prompt-Shortener assistant that summarizes the given prompt exactly as per instructions.
		A prompt is attached. Shorten it. Dont include 'According to', 'As an AI model', 'Here it is', 'Here is the shortened prompt', etc.
		Your response must be shorter than the given prompt.
		The original context question from the user should be preserved in the prompt. 
	'''

	promptTemplate = ChatPromptTemplate.from_messages([
		# ('system', shortener_prompt),
		('user', shortener_prompt + 'Here is the prompt: ```\n{input_prompt}\n```')
	])
	chain = promptTemplate | llm | StrOutputParser()
	result = chain.invoke({ 'input_prompt': input_prompt })
	print(f'Shorten prompt result: \n {result}')

	short_prompt = None
	if len(result) < len(input_prompt):
		short_prompt = result
		print(f'Shorter prompt: {short_prompt}')
	else:
		short_prompt = input_prompt

	cleaned_prompt = clean_prompt(short_prompt, llm)
	output_token_count = llm.get_num_tokens(cleaned_prompt)
	percent_saved = (input_token_count - output_token_count) / input_token_count * 100
	percent_saved = round(percent_saved, 2)
	return cleaned_prompt, input_token_count, output_token_count, percent_saved


supported_image_formats = ['jpg', 'jpeg', 'png']

def convert_to_base64(file):
	# get format and check whether it is supported
	if file.name:
		format = file.name.split('.')[-1]
		if format not in supported_image_formats:
			raise ValueError(f'Unsupported image format: {format}')

	buffered = io.BytesIO()
	# if file is a string
	if isinstance(file, str):
		pil_image = Image.open(file, formats=supported_image_formats)
		pil_image = pil_image.convert('RGB')
		pil_image.save(buffered, format="JPEG")
	elif isinstance(file, Image.Image):
		file.save(buffered, format="JPEG")
	elif isinstance(file, bytes):
		buffered.write(file)
	elif hasattr(file, 'getbuffer'):
		buffered.write(file.getbuffer())
	else:
		raise ValueError('Unsupported image type')
	image_buffer = buffered.getvalue()
	img_str = base64.b64encode(image_buffer).decode("utf-8")
	return img_str

def convert_list_to_base64(files):
	return [convert_to_base64(file) for file in files]


def generate_synthetic_data(llm, topic, rows, fields, examples, data_save_path=None, force=False):
	'generates synthetic data using LLM'

	if not data_save_path:
		data_save_path = os.path.join(datasets_dir, f'{topic}_synthetic_data.csv')
	if not force and os.path.exists(data_save_path):
		return pd.read_csv(data_save_path)

	OPENAI_TEMPLATE = PromptTemplate(input_variables=['example'], template='{example}')
	prompt_template = FewShotPromptTemplate(
		prefix=SYNTHETIC_FEW_SHOT_PREFIX,
		examples=examples,
		suffix=SYNTHETIC_FEW_SHOT_SUFFIX,
		input_variables=fields,
		example_prompt=OPENAI_TEMPLATE,
	)

	generator = SyntheticDataGenerator(llm=llm, template=prompt_template)
	extra_prompt = 'include columns in csv text in triple quotes - ```. \n' \
		f'fields: {", ".join(fields)}. \n' \
		f'Make sure it resembles a real {topic} dataset. \n' \
		f'row count expected: {rows}.'
	data = generator.generate(subject=topic, runs=1, extra=extra_prompt)
	response = data[0]  # read first message, as runs=1
	# get text in quotes ```
	response = response[response.find('```')+3:response.rfind('```')]
	response = response.strip()

	try:
		df = pd.read_csv(io.StringIO(response))
		df.sort_values('date', inplace=True)
		df.reset_index(drop=True, inplace=True)  # sort by date
		df.to_csv(data_save_path, index=False)
		return df
	except Exception as e:
		print('No data generated')
		print(e)


def extract_quotes(text):
	# return the quoted answer which is quoted with ` or ```
	# by extracting from the answer
	if '```' in text:
		text = text.split('```')[1]
	elif '`' in text:
		text = text.split('`')[1]
	return text


