#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import Dataset
import numpy as np
import torch

# Function to split dataset into train, validation, and test sets
def split_dataset(dataset):
    """
    Splits the dataset into train, validation, and test sets with stratification.
    """
    print("Splitting dataset into train, validation, and test sets...")
    
    # Split dataset into train+validation and test
    train_val_df, test_df = train_test_split(
        dataset,
        test_size=0.2,
        stratify=dataset['Converted'],
        random_state=42
    )
    
    # Split train_val_df into train and validation
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=0.25,
        stratify=train_val_df['Converted'],
        random_state=42
    )
    
    # Display class distribution in each set
    print(f"Train set: {len(train_df)} samples (Converted=0: {len(train_df[train_df['Converted'] == 0])}, Converted=1: {len(train_df[train_df['Converted'] == 1])})")
    print(f"Validation set: {len(val_df)} samples (Converted=0: {len(val_df[val_df['Converted'] == 0])}, Converted=1: {len(val_df[val_df['Converted'] == 1])})")
    print(f"Test set: {len(test_df)} samples (Converted=0: {len(test_df[test_df['Converted'] == 0])}, Converted=1: {len(test_df[test_df['Converted'] == 1])})")
    
    return train_df, val_df, test_df

# Split dataset into train, validation, and test
dataset = pd.read_csv("merged.csv")
print(f"Dataset loaded: {len(dataset)} rows")
train_df, val_df, test_df = split_dataset(dataset)

def tokenize_datasets(train_df, test_df, val_df):
    print("Initializing tokenizer and preparing datasets...")
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

    # Function for tokenization and adding the label
    def tokenize_function(examples):
        tokens = tokenizer(examples['Summary'], padding='max_length', truncation=True)
        tokens['label'] = examples['Converted']  # Associate the 'Converted' column as the label
        return tokens

    # Convert the DataFrames to Hugging Face Datasets
    train_dataset = Dataset.from_pandas(train_df[['Summary', 'Converted']])
    test_dataset = Dataset.from_pandas(test_df[['Summary', 'Converted']])
    val_dataset = Dataset.from_pandas(val_df[['Summary', 'Converted']])

    # Apply tokenization separately for training, test, and validation sets
    print("Tokenizing train dataset...")
    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
    print("Tokenizing test dataset...")
    tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)
    print("Tokenizing validation dataset...")
    tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True)
    
    print("Datasets tokenized successfully.")
    return tokenized_train_dataset, tokenized_test_dataset, tokenized_val_dataset, tokenizer

tokenized_train_dataset, tokenized_test_dataset, tokenized_val_dataset, tokenizer = tokenize_datasets(train_df, test_df, val_df)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def compute_metrics(eval_preds):
    """
    Compute accuracy and F1 metrics using scikit-learn.
    """
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)  # Convert logits to predicted labels
    
    print("Calculating metrics...")
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    print(f"Metrics calculated: Accuracy = {acc}, F1 = {f1}")
    
    return {"accuracy": acc, "f1": f1}

def evaluate_model_without_training(test_dataset, tokenizer):
    print("Initializing model for evaluation without training...")
    model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy='epoch',
        per_device_eval_batch_size=16,
        num_train_epochs=1,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    print("Evaluating model on test dataset before training...")
    test_results = trainer.evaluate()
    print("Evaluation complete.")
    print(f"Test Accuracy: {test_results['eval_accuracy']:.4f}")
    print(f"Test F1 Score: {test_results['eval_f1']:.4f}")
    
    return test_results

test_results = evaluate_model_without_training(tokenized_test_dataset, tokenizer)

def train_and_evaluate_model(train_dataset, val_dataset, test_dataset, tokenizer):
    print("Initializing model for training...")
    model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    print("Training the model...")
    trainer.train()
    print("Training complete.")

    print("Evaluating the model on the test set...")
    test_results = trainer.evaluate(test_dataset)
    print(f"Test Accuracy: {test_results['eval_accuracy']:.4f}")
    print(f"Test F1 Score: {test_results['eval_f1']:.4f}")
    
    return model, test_results, trainer

model, test_results, trainer = train_and_evaluate_model(tokenized_train_dataset, tokenized_val_dataset, tokenized_test_dataset, tokenizer)


# In[ ]:




