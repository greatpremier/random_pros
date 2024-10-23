'''from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import torch
from html_data import html_data

# Step 1: Prepare the Dataset
# Example HTML data

# Tokenize the HTML data
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token

# Create input-output pairs
train_data = [
    {
        "input": "Generate html code for a webpage with title and paragraph",
        "output": html
    } for html in html_data
]

# Tokenize the input-output pairs
def tokenize_function(example):
    # Format the input for GPT
    input_text = example['input'] + " " + example['output']  # Concatenate input and output
    return tokenizer(input_text, padding="max_length", truncation=True, return_tensors='pt', max_length=512)

# Tokenize all examples
tokenized_data = [tokenize_function(example) for example in train_data]

# Extract inputs and labels
inputs = torch.cat([data['input_ids'] for data in tokenized_data])
labels = inputs.clone()  # For GPT, labels are the same as inputs for training

# Split the data into training and validation sets
train_inputs, val_inputs, train_labels, val_labels = train_test_split(inputs, labels, test_size=0.2)

# Step 2: Fine-tune the Model
# Load the model
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir='./logs',
)

# Create a custom dataset class
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {
            'input_ids': self.inputs[idx],
            'labels': self.labels[idx]
        }

# Create datasets
train_dataset = CustomDataset(train_inputs, train_labels)
val_dataset = CustomDataset(val_inputs, val_labels)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Start training
trainer.train()

# Save the model and tokenizer
model.save_pretrained('./results')
tokenizer.save_pretrained('./results')'''


from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import load_dataset  # Use datasets to load Common Crawl
import torch

# Step 1: Load the Common Crawl Dataset via Streaming
dataset = load_dataset("c4", "en", split='train', streaming=True)

# Initialize the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Step 2: Tokenization function
def tokenize_function(examples):
    input_text = examples['text']  # Extract the text from the dataset
    return tokenizer(input_text, padding="max_length", truncation=True, return_tensors='pt', max_length=512)

# Step 3: Data preprocessing - Streaming tokenized data
def preprocess_data(dataset, tokenizer):
    for example in dataset:  # Streaming through the dataset one batch at a time
        tokenized_input = tokenize_function(example)
        input_ids = tokenized_input['input_ids']
        yield {
            'input_ids': input_ids,
            'labels': input_ids.clone(),  # GPT is an autoregressive model, so labels are the same as input
        }

# Step 4: Define training arguments with `max_steps`
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,               # Still required for some internal metrics
    per_device_train_batch_size=4,     # Your batch size per GPU
    save_steps=10_000,
    save_total_limit=2,
    logging_dir='./logs',
    logging_steps=500,
    dataloader_drop_last=True,         # Drop last incomplete batch to prevent shape mismatch
    max_steps=20_000,                  # Set a fixed number of training steps (for example, 20,000 steps)
)

# Step 5: Define a custom dataset class
class StreamingDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __iter__(self):
        return preprocess_data(self.dataset, self.tokenizer)

# Step 6: Prepare the streaming dataset
train_dataset = StreamingDataset(dataset, tokenizer)

# Step 7: Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

# Start training the model
trainer.train()

# Save the model and tokenizer
model.save_pretrained('./results')
tokenizer.save_pretrained('./results')
