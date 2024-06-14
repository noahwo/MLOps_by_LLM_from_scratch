import os
import json
import ast
import logging
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(filename="data_engineering.log", level=logging.INFO)

# Define system prompt
system_prompt = """
# CONTEXT #
You are an expert in Tiny Machine Learning (TinyML), highly skilled in the workflow, tools, techniques, and best practices of TinyML operations. Your expertise extends to hardware, including microcontrollers. You will be asked questions regarding various phases, for example, data engineering, model designing, model evaluation, etc, of TinyMLOps and may need to generate code to execute corresponding tasks, for example, data cleaning, model training code, etc.
# OBJECTIVE #
I want to train a model to {purpose}. Now analyze the dataset I uploaded to give practical suggestions in sequential order to do data engineering based on the inspirations you get from this dataset.
# RESPONSE FORMAT #
Keep the answer short and concise. Do not add a title and summary text, question or conclusion in your answer. The output format should be in JSON objects like: {{"suggestion_name":"short explanation"}}
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


# Create a function to get user input
def get_user_input():
    purpose = input("The model is trained for...? ")
    dataset_path = input("Please enter the dataset path: ")
    return purpose, dataset_path


# Create a function to generate the suggestion table
def generate_suggestion_table(purpose):
    llm = OpenAI(temperature=0.7)
    prompt = PromptTemplate(
        input_variables=["purpose"],
        template=system_prompt,
    )
    suggestion_table = json.loads(llm(prompt.format(purpose=purpose)))
    return suggestion_table


# Create a function to execute the code snippet safely
def execute_code(code):
    try:
        ast.literal_eval(code)
        return None
    except Exception as e:
        return str(e)


# Create a function to prompt the LLM for code snippets
def prompt_llm(user_prompt):
    llm = OpenAI(temperature=0.7)
    response = llm(user_prompt)
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
        response = prompt_llm(user_prompt)
        code = response.split("```python")[1].split("```")[0]
    return error


# Create a main function to orchestrate the data engineering process
def main():
    purpose, dataset_path = get_user_input()
    suggestion_table = generate_suggestion_table(purpose)

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


# Run the main function
if __name__ == "__main__":
    main()
