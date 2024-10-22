from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import torch

# Step 1: Prepare the Dataset
# Example HTML data
html_data = [
    """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Sample Page 1</title>
    </head>
    <body>
        <h1>Welcome to Sample Page 1</h1>
        <p>This is a paragraph in sample page 1.</p>
    </body>
    </html>
    """,
    """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Sample Page 2</title>
    </head>
    <body>
        <h1>Welcome to Sample Page 2</h1>
        <p>This is a paragraph in sample page 2.</p>
    </body>
    </html>
    """
]

# Tokenize the HTML data
tokenizer = T5Tokenizer.from_pretrained('t5-small')

def tokenize_function(examples):
    return tokenizer(examples, padding="max_length", truncation=True, return_tensors='pt')

# Create input-output pairs
train_data = [
    {
        "input": "Generate html code for a webpage with title and paragraph",
        "output": html
    } for html in html_data
]

# Tokenize the input-output pairs
inputs = [tokenize_function(example['input'])['input_ids'] for example in train_data]
outputs = [tokenize_function(example['output'])['input_ids'] for example in train_data]

# Convert lists to tensors
inputs = torch.cat(inputs)
outputs = torch.cat(outputs)

# Split the data into training and validation sets
train_inputs, val_inputs, train_outputs, val_outputs = train_test_split(inputs, outputs, test_size=0.2)

# Step 2: Fine-tune the Model
# Load the model
model = T5ForConditionalGeneration.from_pretrained('t5-small')

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
)

# Create a custom dataset class
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {
            'input_ids': self.inputs[idx],
            'labels': self.outputs[idx]
        }

# Create datasets
train_dataset = CustomDataset(train_inputs, train_outputs)
val_dataset = CustomDataset(val_inputs, val_outputs)

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
tokenizer.save_pretrained('./results')