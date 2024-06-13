## TinyML Lifecycle Management By LLMs

The current experiment (based on which the programme will be written) is using a spam email dataset ([Link](https://www.kaggle.com/datasets/abdallahwagih/spam-emails/data)) to train a spam classfier. The used techniques and methods are based on GPT-4o's suggestions, and the code which deals with data/model is generated by GPT-4o as well.

### 1. Data Engineering

The processed is fully inspired and generated by GPT-4o. It successfully suggested and achivened the following steps:

```json
{
  "remove_duplicates": "Check and remove duplicate messages to avoid redundancy.",
  "handle_class_imbalance": "Use techniques such as oversampling or undersampling to balance the number of ham and spam messages.",
  "text_preprocessing": "Perform text preprocessing steps like lowercasing, removing punctuation, stopwords, and stemming/lemmatization on the 'Message' column.",
  "feature_extraction": "Convert text data into numerical data using methods like TF-IDF or word embeddings.",
  "train_test_split": "Split the dataset into training and testing sets to evaluate model performance.",
  "label_encoding": "Encode the 'Category' column (ham/spam) into numerical labels for model training."
}
```

The whole detailed process and prompts are recorded in [this markdown file](./Tenantive-prompts-and-experiment-processes.md).

Next step:
-> implement code to automate this whole manual process (implement data engineering section)
-> explore model design and training following similar process

Should be way faster when the direction is clear.
