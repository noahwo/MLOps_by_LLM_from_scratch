# %% Set up the environment
import os
import json
import ast
import logging
from typing import List
import pandas as pd

# from langchain.llms import OpenAI
from langchain_community.llms import OpenAI
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PipelinePromptTemplate, PromptTemplate
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.documents import Document

# detect spam emails
# ./data/spam.csv

# Load environment variables from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME")
os.environ["LANGCHAIN_API_KEY"] = str(os.getenv("LANGCHAIN_API_KEY"))
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "default"

# Set up logging
logging.basicConfig(filename="data_engineering.log", level=logging.INFO)


"""Prompt templates definitions"""
# Define system prompt

full_prompt = PromptTemplate.from_template(
    """{system_prompt}

    {user_prompt}
    
    {dataset_summary}
    """
)
system_prompt = PromptTemplate.from_template(
    """
You are an expert in Tiny Machine Learning (TinyML), highly skilled in the workflow, tools, techniques, and best practices of TinyML operations. Your expertise extends to hardware, including microcontrollers. You will be asked questions regarding various phases, for example, data engineering, model designing, model evaluation, etc, of TinyMLOps and may need to generate code to execute corresponding tasks, for example, data cleaning, model training code, etc.
"""
)

user_prompt_template_0 = """
# OBJECTIVE #
I want to train a model to {purpose}. Now analyze the dataset I uploaded to give practical suggestions in sequential order to do data engineering based on the inspirations you get from this dataset.
# RESPONSE FORMAT #
Keep the answer short and concise. Do not add a title and summary text, question or conclusion in your answer. The output format should be in JSON objects containing key-value pairs where key is operation name (all lowercase, connected by _), and value is the operation description. output should not contain any char except JSON valid chars, no line change char etc..
"""

# Define user prompt templates
user_prompt_template_1 = """
{{
  "task": {{
    "target_goal": {{
      "{operation_n}": "{operation_n_explanation}",
    }},
    "task_requirements": "Write me practical code to implement the operation.",
    "dataset_path": "{dataset_path}",
    "processing_already_applied": [],
    "format": "output only a code block; report any file and directory structure change caused by your output code; reply similarly to the format of JSON to specify dataset path after change"
  }}
}}
"""

user_prompt_template_2 = """
{{
  "executed_code": "{executed_code}",
  "caused_error": "{error_info}",
  "task": "Regenerate the last task to avoid this error."
}}
"""
suggestion_json_schema = {
    "title": "suggestion_table",
    "description": "suggested operations to perform on the dataset",
    "type": "object",
    "properties": {
        "suggestion_name_1": {
            "type": "string",
            "description": "first suggested operation",
        },
        "suggestion_name_2": {
            "type": "string",
            "description": "second suggested operation",
        },
        "suggestion_name_n": {
            "type": "string",
            "description": "n-th suggested operation",
        },
    },
}
dataset_summary_prompt_template = """The dataset summary information is as follows: {dataset_summary}
"""


# def dataset_loader(dataset_path):
#     loader = CSVLoader(file_path=dataset_path)
#     return loader.load()


def dataset_summary(dataset_path):
    """Return the summary of the dataset as a string to povide inspirations to LLM for data processing suggestion generation."""
    dataframe = pd.read_csv(dataset_path)
    return str(
        {
            "Dataset shape": dataframe.shape,
            "Basic Statistics": dataframe.describe(),
        }
    )


# # %% TEst the dataset loader
# purpose = "detect spam emails"
# dataset_path = "./data/spam.csv"


# %%
def generate_suggestion_table(
    purpose: str,  # The purpose of the model input by user
    dataset_path: str,  # Path to the dataset provided by the user
):
    """Returns a suggestion table based on the dataset meta-info."""
    dataset_summary_str = dataset_summary(dataset_path)
    llm = ChatOpenAI(api_key=OPENAI_API_KEY, model_name=OPENAI_MODEL_NAME)
    pipeline_prompt_0 = PipelinePromptTemplate(
        final_prompt=full_prompt,
        pipeline_prompts=[
            ("system_prompt", system_prompt),
            ("user_prompt", PromptTemplate.from_template(user_prompt_template_0)),
            (
                "dataset_summary",
                PromptTemplate.from_template(dataset_summary_prompt_template),
            ),
        ],
    )

    prompt = pipeline_prompt_0.format(
        purpose=purpose,
        dataset_summary=dataset_summary_str,
    )

    returnable = llm.invoke(prompt)

    return json.loads(returnable.content)


# %%


# Create a function to get user input
def get_user_input():
    purpose = input("The model is trained for...? ")
    dataset_path = input("Please enter the dataset path: ")
    return purpose, dataset_path


# Create a function to execute the code snippet safely
def execute_code(code):
    try:
        ast.literal_eval(code)
        return None
    except Exception as e:
        return str(e)


# Create a function to prompt the LLM for code snippets
def prompt_llm(user_prompt):
    llm = ChatOpenAI(api_key=OPENAI_API_KEY, model_name=OPENAI_MODEL_NAME)

    response = llm.invoke(user_prompt)
    return response


# Create a function to handle retries
def retry_code_execution(code, max_retries=2):
    for _ in range(max_retries):
        error = execute_code(code)
        if not error:
            return None
        logging.error(f"Error: {error}")
        user_prompt = user_prompt_template_2.format(
            executed_code=code, error_info=error
        )
        response = prompt_llm.invoke(user_prompt)
        code = response.split("```python")[1].split("```")[0]
    return error


"""The code below belongs to main() function"""
# %%
# Create a main function to orchestrate the data engineering process
# def main():
# purpose, dataset_path = get_user_input()
purpose = "detect spam emails"
dataset_path = "./data/spam.csv"
suggestion_table = generate_suggestion_table(purpose, dataset_path)
# %%
print(suggestion_table)
# %%
processing_already_applied = []

for operation_n, operation_n_explanation in suggestion_table.items():
    user_prompt = user_prompt_template_1.format(
        operation_n=operation_n,
        operation_n_explanation=operation_n_explanation,
        dataset_path=dataset_path,
    )
    response = prompt_llm(user_prompt)

    code_snippet = response.split("```python")[1].split("```")[0]
    error = retry_code_execution(code_snippet)

    if error:
        logging.error(f"Error: {error}")
        break

    processing_already_applied.append(operation_n)
    dataset_path = json.loads(response.split("```json")[1].split("```")[0])[
        "dataset_path"
    ]

logging.info("Data engineering process completed.")
"""The code above belongs to main() function"""


# %%
# Run the main function
# if __name__ == "__main__":
#     main()
