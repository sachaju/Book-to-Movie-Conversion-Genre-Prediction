#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import Dataset
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc



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
        evaluation_strategy='epoch',  # Valuta il modello alla fine di ogni epoca
        logging_dir='./logs',  # Directory per salvare i log
        logging_steps=100,  # Registra ogni 100 passi
        save_steps=500,  # Salva il modello ogni 500 passi
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
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
        evaluation_strategy='epoch',  # Valuta il modello alla fine di ogni epoca
        logging_dir='./logs',  # Directory per salvare i log
        logging_steps=10,  # Registra ogni 100 passi
        save_steps=500,  # Salva il modello ogni 500 passi
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

# Function to plot the learning curve
def plot_learning_curve(trainer):
    # Extract logs from the trainer (history of metrics during training)
    logs = trainer.state.log_history
    
    # Collect training loss and validation accuracy values
    loss_values = [log['loss'] for log in logs if 'loss' in log]
    eval_accuracy_values = [log['eval_accuracy'] for log in logs if 'eval_accuracy' in log]

    # Check if we have obtained values for loss and accuracy
    if not loss_values:
        print("No loss recorded.")
    if not eval_accuracy_values:
        print("No validation accuracy recorded.")
    
    # Training loss is an indicator of how well the model is fitting to the training data.
    # A decrease in training loss means the model is learning to reduce the error during training.
    # Validation accuracy, on the other hand, measures how well the model generalizes to unseen data, i.e., how accurate it is on the validation set.
    # These metrics help us understand if the model is overfitting or underfitting.

    plt.figure(figsize=(12, 6))

    # Plot of training loss
    plt.subplot(1, 2, 1)
    if loss_values:
        plt.plot(loss_values, label='Training Loss')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.legend()

    # Plot of validation accuracy
    plt.subplot(1, 2, 2)
    if eval_accuracy_values:
        plt.plot(eval_accuracy_values, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Validation Accuracy Curve')
        plt.legend()

    plt.tight_layout()
    plt.show()

# After training, call this function to plot the graph
plot_learning_curve(trainer)


def plot_confusion_matrix(trainer, test_dataset):
    # Extract predictions from the model
    predictions, labels, _ = trainer.predict(test_dataset)
    preds = np.argmax(predictions, axis=-1)  # Final predictions
    
    # Compute the confusion matrix
    cm = confusion_matrix(labels, preds)
    
    # The confusion matrix is crucial for understanding the quality of the model's predictions.
    # It shows the number of correct and incorrect predictions for each class.
    # From the confusion matrix, we can derive metrics such as precision, recall, and accuracy.
    # It also helps identify any model bias (e.g., if one class is incorrectly predicted more frequently).

    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()

# After training, call this function
plot_confusion_matrix(trainer, tokenized_test_dataset)


def print_classification_report(trainer, test_dataset):
    # Extract predictions
    predictions, labels, _ = trainer.predict(test_dataset)
    preds = np.argmax(predictions, axis=-1)  # Final predictions
    
    # Print the classification report
    report = classification_report(labels, preds, target_names=['Class 0', 'Class 1'])
    # The classification report includes metrics like:
    # - Precision: Measures the ability of the model to not label negative examples as positive. It is important when false positives are costly.
    # - Recall: Measures the ability of the model to identify all positive examples. It is important when false negatives are problematic.
    # - F1-score: The harmonic mean of precision and recall. It is particularly useful when a balance between the two metrics is needed.
    print("Classification Report:\n", report)

# After training, call this function
print_classification_report(trainer, tokenized_test_dataset)


def plot_roc_curve(trainer, test_dataset):
    # Extract predictions and compute the ROC curve
    predictions, labels, _ = trainer.predict(test_dataset)
    preds = np.argmax(predictions, axis=-1)  # Final predictions
    fpr, tpr, _ = roc_curve(labels, predictions[:, 1])  # Using the probability of class 1
    roc_auc = auc(fpr, tpr)
    
    # The ROC curve (Receiver Operating Characteristic) shows the trade-off between the false positive rate (FPR) and the true positive rate (TPR).
    # The area under the curve (AUC) provides a summary of the model's ability to distinguish between classes.
    # An AUC closer to 1 indicates a better model, while an AUC close to 0.5 indicates a model making random predictions.

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()

# After training, call this function
plot_roc_curve(trainer, tokenized_test_dataset)


def show_predictions(trainer, test_dataset, num_examples=5):
    # Extract predictions
    predictions, labels, _ = trainer.predict(test_dataset)
    preds = np.argmax(predictions, axis=-1)  # Final predictions
    
    # Select a random number of examples and display the predictions
    for i in range(num_examples):
        print(f"Summary: {test_dataset[i]['Summary']}")
        print(f"True Label: {test_dataset[i]['Converted']}")
        print(f"Predicted Label: {preds[i]}")
        print('-' * 50)

# After training, call this function
show_predictions(trainer, tokenized_test_dataset)

