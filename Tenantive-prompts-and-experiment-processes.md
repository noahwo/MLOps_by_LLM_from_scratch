[toc]

## System prompt

> You are an expert in Tiny Machine Learning (TinyML), highly skilled in the workflow, tools, techniques, and best practices of TinyML operations. Your expertise extends to hardware, including microcontrollers. You will be asked questions regarding various phases, for example, data engineering, model designing, model evaluation, etc, of TinyMLOps and may need to generate code to execute corresponding tasks, for example, data cleaning, model training code, etc.

## Example - suggestions and code for data engineering

### Prompt 1

> \# CONTEXT #
> You are an expert in Tiny Machine Learning (TinyML), highly skilled in the workflow, tools, techniques, and best practices of TinyML operations. Your expertise extends to hardware, including microcontrollers. You will be asked questions regarding various phases, for example, data engineering, model designing, model evaluation, etc, of TinyMLOps and may need to generate code to execute corresponding tasks, for example, data cleaning, model training code, etc.
> \# OBJECTIVE #
> I want to train a model to detect spam emails. now analyse the dataset I uploaded to give practical suggestions in sequential order to do data engineering based on the inspirations you get from this dataset.
> \# RESPONSE FORMAT #
> Keep the answer short and concise. do not add a title and summary text, question or conclusion in your answer. the output format should be in JSON objects like: {"suggestion_name":"short explanation"}

### Answer 1

```JSON
{
  "remove_duplicates": "Check and remove duplicate messages to avoid redundancy.",
  "handle_class_imbalance": "Use techniques such as oversampling or undersampling to balance the number of ham and spam messages.",
  "text_preprocessing": "Perform text preprocessing steps like lowercasing, removing punctuation, stopwords, and stemming/lemmatization on the 'Message' column.",
  "feature_extraction": "Convert text data into numerical data using methods like TF-IDF or word embeddings.",
  "train_test_split": "Split the dataset into training and testing sets to evaluate model performance.",
  "label_encoding": "Encode the 'Category' column (ham/spam) into numerical labels for model training."
}

```

### prompt 2

```json
{
  "task": {
    "target_goal": {
      "remove_duplicates": "Check and remove duplicate messages to avoid redundancy."
    },
    "task_requirements": "Write me practical code to implement the operation.",
    "dataset_path": "./data/spam.csv",
    "processing_already_applied": [],
    "format": "output only a code block; report any file and directory structure change caused by your output code; replay similarly to the format of JSON to specify dataset path after change"
  }
}
```

### answer 2

```python
import pandas as pd

# Load the dataset
file_path = './data/spam.csv'
data = pd.read_csv(file_path)

# Remove duplicate messages
data = data.drop_duplicates(subset='Message')

# Save the cleaned dataset
cleaned_file_path = './data/spam_cleaned.csv'
data.to_csv(cleaned_file_path, index=False)
```

```json
{ "dataset_path": "./data/spam_cleaned.csv" }
```

### prompt 3

```json
{
  "task": {
    "target_goal": {
      "handle_class_imbalance": "Use techniques such as oversampling or undersampling to balance the number of ham and spam messages."
    },
    "task_requirements": "Write me practical code to implement the operation.",
    "dataset_path": "./data/spam_cleaned.csv",
    "processing_already_applied": ["remove_duplicates"],
    "format": "output only a code block; report any file and directory structure change caused by your output code; replay similarly to the format of JSON to specify dataset path after change"
  }
}
```

### answer 3

```python
import pandas as pd
from sklearn.utils import resample

# Load the cleaned dataset
file_path = './data/spam_cleaned.csv'
data = pd.read_csv(file_path)

# Separate majority and minority classes
ham = data[data.Category == 'ham']
spam = data[data.Category == 'spam']

# Upsample minority class (spam)
spam_upsampled = resample(spam,
                          replace=True,    # sample with replacement
                          n_samples=len(ham),  # match number in majority class
                          random_state=42)  # reproducible results

# Combine majority class with upsampled minority class
upsampled_data = pd.concat([ham, spam_upsampled])

# Save the balanced dataset
balanced_file_path = './data/spam_balanced.csv'
upsampled_data.to_csv(balanced_file_path, index=False)
```

```json
{
  "dataset_path": "./data/spam_balanced.csv"
}
```

### prompt 4

```json
{
  "task": {
    "target_goal": {
      "text_preprocessing": "Perform text preprocessing steps like lowercasing, removing punctuation, stopwords, and stemming/lemmatization on the 'Message' column."
    },
    "task_requirements": "Write me practical code to implement the operation.",
    "dataset_path": "./data/spam_cleaned.csv",
    "processing_already_applied": [
      "remove_duplicates",
      "handle_class_imbalance"
    ],
    "format": "output only a code block; report any file and directory structure change caused by your output code; replay similarly to the format of JSON to specify dataset path after change"
  }
}
```

### answer 4

```python
import string
import pandas as pd
from sklearn.utils import resample
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Ensure you have downloaded the necessary NLTK data files.
import nltk
nltk.download('punkt')
nltk.download('stopwords')

# Load the balanced dataset
file_path = './data/spam_balanced.csv'
data = pd.read_csv(file_path)

# Initialize stemmer
stemmer = PorterStemmer()

# Function for text preprocessing
def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize
    words = word_tokenize(text)
    # Remove stopwords and stem
    words = [stemmer.stem(word) for word in words if word not in stopwords.words('english')]
    # Join words back to string
    return ' '.join(words)

# Apply preprocessing to the 'Message' column
data['Message'] = data['Message'].apply(preprocess_text)

# Save the preprocessed dataset
preprocessed_file_path = './data/spam_preprocessed.csv'
data.to_csv(preprocessed_file_path, index=False)
```

```json
{
  "dataset_path": "./data/spam_preprocessed.csv"
}
```

### Issue 1 - package missing

Package `nltk`m missing, I manually installed it then.

### prompt 5

```json
{
  "task": {
    "target_goal": {
      "feature_extraction": "Convert text data into numerical data using methods like TF-IDF or word embeddings."
    },
    "task_requirements": "Write me practical code to implement the operation.",
    "dataset_path": "./data/spam_preprocessed.csv",
    "processing_already_applied": [
      "remove_duplicates",
      "handle_class_imbalance",
      "text_preprocessing"
    ],
    "format": "output only a code block; report any file and directory structure change caused by your output code; replay similarly to the format of JSON to specify dataset path after change"
  }
}
```

### answer 5

```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Load the preprocessed dataset
file_path = './data/spam_preprocessed.csv'
data = pd.read_csv(file_path)

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=5000)  # Limiting to 5000 features for efficiency

# Fit and transform the 'Message' column
X = vectorizer.fit_transform(data['Message'])

# Save the TF-IDF features and the vectorizer
features_file_path = './data/spam_features.pkl'
vectorizer_file_path = './data/tfidf_vectorizer.pkl'

with open(features_file_path, 'wb') as f:
    pickle.dump(X, f)

with open(vectorizer_file_path, 'wb') as f:
    pickle.dump(vectorizer, f)
```

```json
{
  "dataset_path_features": "./data/spam_features.pkl",
  "vectorizer_path": "./data/tfidf_vectorizer.pkl"
}
```

### Issue 2 - error

```
{“executed_code”:”import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Load the preprocessed dataset
file_path = './data/spam_preprocessed.csv'
data = pd.read_csv(file_path)

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=5000)  # Limiting to 5000 features for efficiency

# Fit and transform the 'Message' column
X = vectorizer.fit_transform(data['Message'])

# Save the TF-IDF features and the vectorizer
features_file_path = './data/spam_features.pkl'
vectorizer_file_path = './data/tfidf_vectorizer.pkl'

with open(features_file_path, 'wb') as f:
    pickle.dump(X, f)

with open(vectorizer_file_path, 'wb') as f:
    pickle.dump(vectorizer, f)”,

“caused_error”:”Traceback (most recent call last):
  File "/Users/hann/Projects/MCUrelated/MLOps_by_LLM_from_scratch/datapreprocess.py", line 15, in <module>
    X = vectorizer.fit_transform(data["Message"])
  File "/Users/hann/anaconda3/envs/datasci/lib/python3.10/site-packages/sklearn/feature_extraction/text.py", line 2139, in fit_transform
    X = super().fit_transform(raw_documents)
  File "/Users/hann/anaconda3/envs/datasci/lib/python3.10/site-packages/sklearn/base.py", line 1152, in wrapper
    return fit_method(estimator, *args, **kwargs)
  File "/Users/hann/anaconda3/envs/datasci/lib/python3.10/site-packages/sklearn/feature_extraction/text.py", line 1389, in fit_transform
    vocabulary, X = self._count_vocab(raw_documents, self.fixed_vocabulary_)
  File "/Users/hann/anaconda3/envs/datasci/lib/python3.10/site-packages/sklearn/feature_extraction/text.py", line 1276, in _count_vocab
    for feature in analyze(doc):
  File "/Users/hann/anaconda3/envs/datasci/lib/python3.10/site-packages/sklearn/feature_extraction/text.py", line 105, in _analyze
    doc = decoder(doc)
  File "/Users/hann/anaconda3/envs/datasci/lib/python3.10/site-packages/sklearn/feature_extraction/text.py", line 238, in decode
    raise ValueError(
ValueError: np.nan is an invalid document, expected byte or unicode string.”,
“task”:”regenerate the last task to avoid this error”}
```

### answer - issue 2

```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Load the preprocessed dataset
file_path = './data/spam_preprocessed.csv'
data = pd.read_csv(file_path)

# Check for any NaN values in the 'Message' column
data['Message'] = data['Message'].fillna('')

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=5000)  # Limiting to 5000 features for efficiency

# Fit and transform the 'Message' column
X = vectorizer.fit_transform(data['Message'])

# Save the TF-IDF features and the vectorizer
features_file_path = './data/spam_features.pkl'
vectorizer_file_path = './data/tfidf_vectorizer.pkl'

with open(features_file_path, 'wb') as f:
    pickle.dump(X, f)

with open(vectorizer_file_path, 'wb') as f:
    pickle.dump(vectorizer, f)
```

```json
{
  "dataset_path_features": "./data/spam_features.pkl",
  "vectorizer_path": "./data/tfidf_vectorizer.pkl"
}
```

### prompt 6

```json
{
  "task": {
    "target_goal": {
      "train_test_split": "Split the dataset into training and testing sets to evaluate model performance."
    },
    "task_requirements": "Write me practical code to implement the operation.",
    "dataset_path": {
      "dataset_path_features": "./data/spam_features.pkl",
      "vectorizer_path": "./data/tfidf_vectorizer.pkl"
    },
    "processing_already_applied": [
      "remove_duplicates",
      "handle_class_imbalance",
      "text_preprocessing",
      "feature_extraction"
    ],
    "format": "output only a code block; report any file and directory structure change caused by your output code; replay similarly to the format of JSON to specify dataset path after change"
  }
}
```

### answer 6

```python
import pickle
from sklearn.model_selection import train_test_split
import pandas as pd

# Load the TF-IDF features and labels
features_file_path = './data/spam_features.pkl'
vectorizer_file_path = './data/tfidf_vectorizer.pkl'
data_file_path = './data/spam_preprocessed.csv'

with open(features_file_path, 'rb') as f:
    X = pickle.load(f)

# Load the dataset to get the labels
data = pd.read_csv(data_file_path)
y = data['Category'].apply(lambda x: 1 if x == 'spam' else 0)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save the split datasets
train_test_split_path = './data/train_test_split.pkl'
with open(train_test_split_path, 'wb') as f:
    pickle.dump((X_train, X_test, y_train, y_test), f)
```

```json
{
  "train_test_split_path": "./data/train_test_split.pkl"
}
```

### prompt 7

```json
{
  "task": {
    "target_goal": {
      "label_encoding": "Encode the 'Category' column (ham/spam) into numerical labels for model training."
    },
    "task_requirements": "Write me practical code to implement the operation.",
    "dataset_path": {
      "train_test_split_path": "./data/train_test_split.pkl"
    },
    "processing_already_applied": [
      "remove_duplicates",
      "handle_class_imbalance",
      "text_preprocessing",
      "feature_extraction",
      "train_test_split"
    ],
    "format": "output only a code block; report any file and directory structure change caused by your output code; replay similarly to the format of JSON to specify dataset path after change"
  }
}
```

### answer 7

```python
import pickle
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Load the split datasets
train_test_split_path = './data/train_test_split.pkl'

with open(train_test_split_path, 'rb') as f:
    X_train, X_test, y_train, y_test = pickle.load(f)

# Encode the labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Save the encoded labels along with the features
encoded_split_path = './data/encoded_train_test_split.pkl'
with open(encoded_split_path, 'wb') as f:
    pickle.dump((X_train, X_test, y_train_encoded, y_test_encoded), f)

# Save the label encoder for future use
label_encoder_path = './data/label_encoder.pkl'
with open(label_encoder_path, 'wb') as f:
    pickle.dump(label_encoder, f)
```

```json
{
  "encoded_split_path": "./data/encoded_train_test_split.pkl",
  "label_encoder_path": "./data/label_encoder.pkl"
}
```
