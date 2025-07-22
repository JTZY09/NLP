import torch
import json
from datasets import DatasetDict, Dataset
from transformers import (
    TrainingArguments, Trainer, GPT2LMHeadModel, GPT2Tokenizer,
    DataCollatorForLanguageModeling, GPT2Config
)
from sklearn.model_selection import train_test_split

import os
os.environ["WANDB_DISABLED"] = "true"

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# Load the cleaned dataset
file_path = "/content/clash_of_clans_realistic_qa_dataset.csv"  # Ensure you have this file
df = pd.read_csv(file_path)

# Split into 80% training and 20% validation
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Save as separate files
train_df.to_csv("coc_chatbot_train.csv", index=False)
val_df.to_csv("coc_chatbot_validation.csv", index=False)

# Load datasets using Hugging Face's dataset loader
dataset = load_dataset("csv", data_files={
    "train": "coc_chatbot_train.csv",
    "validation": "coc_chatbot_validation.csv"
})


from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the GPT-2 model and tokenizer
model_name = "gpt2-medium"  # You can use "gpt2", "gpt2-large" if needed
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Ensure the tokenizer can handle padding
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    # Concatenate Prompt and Response for each example
    combined_texts = [p + " " + r for p, r in zip(examples["Question"], examples["Response"])]

    # Tokenize inputs
    tokenized_outputs = tokenizer(combined_texts, padding="max_length", truncation=True, max_length=128)

    # Set input_ids as labels for supervised learning
    tokenized_outputs["labels"] = tokenized_outputs["input_ids"].copy()

    return tokenized_outputs

# Apply tokenization with correct batching
tokenized_dataset = dataset.map(tokenize_function, batched=True)

from transformers import TrainingArguments, Trainer

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",  # Evaluate after every epoch
    save_strategy="epoch",  # Save model every epoch
    per_device_train_batch_size=2,  # Adjust based on GPU availability
    per_device_eval_batch_size=2,
    num_train_epochs=3,  # Increase for better fine-tuning
    learning_rate=3e-5,
    weight_decay=0.15,
    logging_dir="./logs",
    logging_steps=50,
    save_total_limit=2,  # Save only the 2 most recent models
    report_to="none"  # Disable logging to external services
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"]
)

# Train the model
trainer.train()

import math

# Retrieve last logged training loss safely
train_loss = None
for log in reversed(trainer.state.log_history):  # Iterate backwards
    if 'loss' in log:
        train_loss = log['loss']
        break  # Stop at the last recorded loss

# Ensure a valid loss value is found
if train_loss is None:
    raise ValueError("No training loss found in the log history.")

# Get validation loss
valid_loss = trainer.evaluate()["eval_loss"]  # Retrieve validation loss

# Compute perplexity safely
perplexity = math.exp(valid_loss) if valid_loss < 10 else float("inf")

# Print results in required format
print(f"\nFinal Training Loss: {train_loss:.4f}")
print(f"Final Validation Loss: {valid_loss:.8f}")
print(f"Model Perplexity: {perplexity:.2f} (Lower is better)")

import shutil
from transformers import GPT2Tokenizer

# Load original GPT-2 tokenizer
model_name = "gpt2-medium"  # Ensure this matches the model you fine-tuned
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Save tokenizer to checkpoint directory
tokenizer.save_pretrained("coc_model\model\results\checkpoint-18360")  # Ensure correct checkpoint





















from transformers import pipeline

# Specify the correct checkpoint path
model_path = "coc_model\model\results\checkpoint-18360"  # Use latest checkpoint

# Load fine-tuned model and tokenizer
qa_pipeline = pipeline("text-generation", model=model_path, tokenizer=model_path)

# Ask a question
question = "Who won vct champions seoul 2024?"
result = qa_pipeline(question, max_length=100)

# Print the generated response
print(result[0]["generated_text"])

# Save the fine-tuned model and tokenizer together
save_directory = "./Model"

# Save model
model.save_pretrained(save_directory)

# Save tokenizer
tokenizer.save_pretrained(save_directory)

print(f"Fine-tuned model and tokenizer saved in: {save_directory}")

from google.colab import drive
import shutil

# Mount Google Drive
drive.mount('/content/drive')

# Define paths
local_model_path = "/content/Model"
drive_model_path = "/content/drive/My Drive/model"

# Copy model to Google Drive
shutil.copytree(local_model_path, drive_model_path)

print(f"Model uploaded to Google Drive at: {drive_model_path}")

from google.colab import drive
drive.mount('/content/drive')