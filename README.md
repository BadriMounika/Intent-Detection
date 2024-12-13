## Title
# Intent-Detection
## Description
This task implements a machine learning model for intent detection, a crucial component in Conversational AI. By leveraging the powerful BERT model, the system can classify user queries into predefined categories such as **EMI, Warranty, Delivery**, and more. The approach ensures contextual understanding of user inputs, enabling accurate and meaningful responses.

# Table of Contents
- [Overview](#Overview)
- [Dataset](#dataset)
- [Dependencies](#dependencies)
- [Structure](#structure)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Evaluation Metrics](#evaluation-metrics)
- [Testing a Custom Query](#testing-a-custom-query)
- [Conclusion](#conclusion)
## Overview
Intent detection is a multi-class text classification task where the system predicts the intent behind a user query. For example:

Query: "What is the EMI option?"

Predicted Intent: **EMI**

This task explores **two approaches** to intent detection:
## 1. Traditional Machine Learning Approaches
Traditional methods like **Naive Bayes, Support Vector Machines (SVM)**, or **Logistic Regression** use manually engineered features such as:

- **TF-IDF** (Term Frequency-Inverse Document Frequency): Converts text into 
  numerical features based on word importance.
- **Bag-of-Words**: Represents text as a collection of word occurrences.
  
### Limitations:
- **Lack of Context**: These methods treat words independently and fail to 
  capture relationships between words in a sentence.
- **Manual Feature Engineering**: Requires extensive effort to preprocess 
  text and extract meaningful features.
- **Performance**: Accuracy is limited when handling complex queries or 
  nuanced intents.
## 2. BERT-Based Approach
BERT (**Bidirectional Encoder Representations from Transformers**) is a pre-trained deep learning model that understands the **context** of words in a sentence by analyzing both the left and right context simultaneously. Fine-tuning BERT for intent detection involves adapting the pre-trained model to classify user queries into specific intents.

## Why BERT?
### 1. Contextual Understanding:

- Unlike traditional methods, BERT captures the relationships between words 
  in a sentence, improving understanding of user queries.
- Example: "Can I pay in installments?" vs. "What is the EMI option?". BERT 
  can distinguish these despite their semantic similarity.
### 2. Pre-Trained Knowledge:

- BERT has been trained on massive datasets, enabling it to generalize 
  better even with limited domain-specific data.
### 3. State-of-the-Art Accuracy:

- BERT outperforms traditional models in tasks like text classification by 
  leveraging its deep, transformer-based architecture.
## Why Not Traditional Methods?
While traditional methods are computationally cheaper and simpler to implement, they fail to handle the complexity of natural language:

- **Context Ignorance**: They rely on isolated word frequencies without 
  understanding sentence structure or meaning.
- **Scalability Issues**: Performance degrades significantly as the number 
  of classes or complexity of queries increases.
  
In contrast, BERT achieves superior accuracy and contextual understanding, making it the preferred choice for this project.

## Dataset
The dataset consists of text queries mapped to corresponding intents. Example intents include:

- **EMI**: Queries related to installment payments.
- **Warranty**: Questions about product warranty details.
- **Delivery**: Queries regarding delivery schedules and options.
- **Returns**: Questions about return policies.
  
The dataset is preprocessed to clean text, tokenize sentences, and encode intent labels.

## Dependencies
To run this project, ensure you have the following Python libraries installed:

- **Transformers**: ``` pip install transformers ```
- **PyTorch**: ``` pip install torch ```
- **Scikit-learn**: ``` pip install scikit-learn ```
- **Pandas**: ``` pip install pandas ```
- **NumPy**: ``` pip install numpy ```
  
Alternatively, install all dependencies using:

```sh
pip install -r requirements.txt
```
## Structure
**1. Data Preprocessing**

**2. Dataset Preparation**

**3. Model Initialization**

**4. Model Training**

**5. Model Evaluation**

**6. Predict Custom Queries**


## Data Preprocessing
- **Text Cleaning**: Remove special characters, extra spaces, and convert text to lowercase.

- **Tokenization**: Break down sentences into words/subwords using BERT's pre-trained tokenizer to convert text into numerical format.

- **Padding and Truncation**: Ensure all text sequences are of the same length by padding or truncating the text.

- **Encoding Labels**: Convert intent labels (e.g., EMI) into numeric values using LabelEncoder.

- **Data Splitting**: Split the data into 80% training and 20% testing sets for model training and evaluation.


## Model Training
The model is fine-tuned using BERT (bert-base-uncased), a transformer-based model pre-trained on vast amounts of text. Key steps:

**1. Preprocessing**: Tokenize the text, pad/truncate sequences, and encode 
     intent labels.
     
**2. Fine-Tuning**: Train the BERT model on the training dataset using the 
     **AdamW optimizer**.
     
**3. Saving the Model**: Save the fine-tuned model and tokenizer for future 
     use.
     
     
To train the model, run:
```sh
python intent_detection.py
```

## Evaluation Metrics
The model’s performance is evaluated using:

- **Accuracy**: Measures the percentage of correct predictions.
- **Precision**: Indicates how many predicted intents are correct.
- **Recall**: Shows how well the model identifies all instances of each 
  intent.
- **F1-Score**: Balances precision and recall for each class.

Example results:
``` sh
Accuracy: 0.50
Classification Report:
               precision    recall  f1-score   support

          EMI       0.60      0.55      0.57        20
      Warranty       0.50      0.60      0.55        15
       Delivery      0.45      0.40      0.42        10
        Return       0.50      0.50      0.50         5
```

## Testing a Custom Query
You can test the model with your own queries using the interactive prediction function:
1. Run the script:
   ```sh
   python intent_detection.py
   ```
2. Enter a query when prompted:
   ```sh
   Enter your query (or 'exit' to quit): What is the warranty on this 
   product?
   Predicted Intent: Warranty
  

## Conclusion

This task demonstrates the potential of BERT for intent detection, achieving moderate accuracy while handling multi-class text classification. Improvements such as **data augmentation** and **hyperparameter** tuning can further enhance performance. The system showcases how deep learning models can empower Conversational AI to better understand and respond to user queries.

