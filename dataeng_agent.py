# %% Set up the environment
import os
import subprocess
import json
import ast
import logging
from typing import List
import pandas as pd
import local_templates
import traceback
import sys

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
logging.basicConfig(level=logging.WARNING)


"""Prompt definitions"""


# %%
full_prompt = PromptTemplate.from_template(local_templates.full_prompt_template)
system_prompt = PromptTemplate.from_template(local_templates.system_prompt_template)
user_prompt_0 = PromptTemplate.from_template(local_templates.user_prompt_template_0)
user_prompt_1 = PromptTemplate.from_template(local_templates.user_prompt_template_1)
user_prompt_2 = PromptTemplate.from_template(local_templates.user_prompt_template_2)
dataset_summary_prompt = PromptTemplate.from_template(
    local_templates.dataset_summary_prompt_template
)
chat_history = """ """
# %%


def update_chat_history(prompt, response_raw):
    """Synthesize a chat history item from the prompt and response."""
    global chat_history
    chat_history = (
        """\n\n# Most recent round of conversation: # \n"""
        + "["
        + f"HumanMessage({str(prompt)}),"
        + f"AIMessage({str(response_raw.content)})"
        + "],\n"
    )


# %%
# %%
def dataset_summary(dataset_path):
    """Return the summary of the dataset as a string to povide inspirations to LLM for data processing suggestion generation."""
    dataframe = pd.read_csv(dataset_path)
    description = "columns        " + str(dataframe.describe())
    return (
        str(
            {
                "Dataset shape": dataframe.shape,
                "Basic Statistics": description,
            }
        )
        + "\nNote: column names are case sensitive, remember that."
    )


# %%
def generate_suggestion_table(
    purpose: str,  # The purpose of the model input by user
    dataset_path: str,  # Path to the dataset provided by the user
    dataset_intro: str,  # Introduction to the dataset provided by the user
):
    """Returns a suggestion table based on the dataset meta-info."""
    dataset_summary_str = dataset_summary(dataset_path)
    llm = ChatOpenAI(api_key=OPENAI_API_KEY, model_name=OPENAI_MODEL_NAME)
    pipeline_prompt_0 = PipelinePromptTemplate(
        final_prompt=full_prompt,
        pipeline_prompts=[
            ("system_prompt", system_prompt),
            ("user_prompt", user_prompt_0),
            (
                "dataset_summary",
                dataset_summary_prompt,
            ),
        ],
    )

    prompt = pipeline_prompt_0.format(
        purpose=purpose,
        dataset_summary=dataset_summary_str,
        dataset_intro=dataset_intro,
    )

    returnable = llm.invoke(prompt)
    update_chat_history(prompt, returnable)
    return json.loads(returnable.content)


# %%


# Create a function to get user input
def get_user_input():
    purpose = input("The model is trained for...? ")
    dataset_path = input("Please enter the dataset path: ")
    dataset_intro = input("Please provide a brief introduction to the dataset: ")
    return purpose, dataset_path, dataset_intro


# Create a function to execute the code snippet safely
def execute_code(code):
    """Decoupled local code execution function."""

    tmp_file = "./tmp/tmp1.py"
    with open(tmp_file, "w") as file:
        file.write(code)

    # Try to execute the script and handle errors
    try:
        result = subprocess.run(["python", tmp_file], capture_output=True, text=True)
        if result.returncode == 0:
            return None
        else:

            # Handle the error
            # logging.error("An error occurred during script execution:")
            # logging.error(result.stderr)

            return result.stderr

    finally:
        # Ensure the script file is deleted regardless of success or failure
        os.remove(tmp_file)


# Create a function to prompt the LLM for code snippets
# def prompt_llm(user_prompt):
#     """Instantiate the ChatOpenAI class and prompt the LLM for code snippets each time."""
#     llm = ChatOpenAI(api_key=OPENAI_API_KEY, model_name=OPENAI_MODEL_NAME)

#     response = llm.invoke(user_prompt)
#     return response.content


# Create a function to handle retries
def retry_code_execution(
    code,
    max_retries=2,
    llm=None,
):
    """General function for local code execution, also responsible for handling re-prompting and re-executing in case of errors."""
    for i in range(max_retries + 1):
        if i == max_retries:
            sys.exit(
                f"An error could not be resolved after {max_retries} retries: \n{error}"
            )
        error = execute_code(code)
        if not error:
            return [None, llm]
        logging.error(f"Error: {error}")
        logging.info(f"Trying to solve the error... Attempt {i + 1}...")
        user_prompt = user_prompt_2.format(executed_code=code, error_info=error)
        response_raw = llm.invoke(str(user_prompt + chat_history))
        response = response_raw.content
        update_chat_history(user_prompt, response_raw)
        code = response.split("```python")[1].split("```")[0]

    return [error, llm]


"""The code below belongs to main() function"""


# %%
# Create a main function to orchestrate the data engineering process
def main():
    # purpose, dataset_path = get_user_input()
    purpose = "detect spam emails"
    dataset_path = "./data/spam.csv"
    dataset_intro = """This dataset contains a collection of emails, categorized into two classes: "Spam" and "Non-Spam" (often referred to as "Ham"). These emails have been carefully curated and labeled to aid in the development of spam email detection models. Whether you are interested in email filtering, natural language processing, or machine learning, this dataset can serve as a valuable resource for training and evaluation."""
    suggestion_table = generate_suggestion_table(purpose, dataset_path, dataset_intro)

    # %%
    processing_already_applied = []
    llm = ChatOpenAI(api_key=OPENAI_API_KEY, model_name=OPENAI_MODEL_NAME)

    for operation_n, operation_n_explanation in suggestion_table.items():
        user_prompt = user_prompt_1.format(
            operation_n=operation_n,
            operation_n_explanation=operation_n_explanation,
            dataset_path=dataset_path,
            list_processing_already_applied=str(processing_already_applied),
        )
        response_raw = llm.invoke(user_prompt + chat_history)
        response = response_raw.content
        update_chat_history(user_prompt, response_raw)
        code_snippet = response.split("```python")[1].split("```")[0]
        returned_list = retry_code_execution(code_snippet, 2, llm)
        llm = returned_list[1]
        if returned_list[0] is not None:
            error = returned_list[0]
            sys.exit(f"Exiting due to failed 2 tries to solve the error: \n{error}")

        processing_already_applied.append(operation_n)
        dataset_path = json.loads(
            response.split("```json")[1].split("```")[0].strip().replace("'", '"')
        )

    logging.info("Data engineering process completed.")


"""The code above belongs to main() function"""


# %%
# Run the main function
if __name__ == "__main__":
    main()
