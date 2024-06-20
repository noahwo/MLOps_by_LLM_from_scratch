full_prompt_template = """{system_prompt}

{user_prompt}

{dataset_summary}"""

system_prompt_template = """ You are an expert in Tiny Machine Learning (TinyML), highly skilled in the workflow, tools, techniques, and best practices of TinyML operations. Your expertise extends to hardware, including microcontrollers. You will be asked questions regarding various phases, for example, data engineering, model designing, model evaluation, etc, of TinyMLOps and may need to generate code to execute corresponding tasks, for example, data cleaning, model training code, etc.
"""


user_prompt_template_0 = """# OBJECTIVE #
I want to train a model to {purpose}. Now analyze the dataset I uploaded to give practical suggestions in sequential order to do data engineering based on the inspirations you get from this dataset. When applying each suggestion, the dataset would be read from the file again, so do not give suggestions like "load_dataset", because that is basic operation contained in every suggestion.

# RESPONSE FORMAT #
# Keep the answer short and concise. Do not add a title and summary text, question or conclusion in your answer. The output format should be in JSON objects containing key-value pairs where key is operation name (all lowercase, connected by _), and value is the operation description. output should not contain any char except JSON valid chars, no line change char etc.."""

user_prompt_template_1 = """
{{
  "task": {{
    "target_goal": {{
      "{operation_n}": "{operation_n_explanation}",
    }},
    "task_requirements": "Write me practical code to implement the operation.",
    "dataset_path": {dataset_path},
    "processing_already_applied": {list_processing_already_applied},
    "format": "output only two code blocks; code in the first block; in the second code block, report any file and directory structure change caused by your output code; replay similarly to the format of JSON to specify dataset path after change. Do not assume things, the code should be clear, accurate and executable, skip any code you are unsure about the detail.",
  }}
}}
"""

user_prompt_template_2 = """{{
    "executed_code": "{executed_code}",
    "caused_error": "{error_info}",
    "task": "Regenerate the last task to avoid this error.",
    "format": "output only two code blocks; code in the first block; in the second code block, report any file and directory structure change caused by your output code; replay similarly to the format of JSON to specify dataset path after change. Do not assume things, the code should be clear, accurate and executable, skip any code you are unsure about the detail.",
}}"""


dataset_summary_prompt_template = """
    Brief introduction to the dataset: {dataset_intro}\n
    The dataset summary information is as follows:\n {dataset_summary}
    """
