from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset

# Load tokenizer and model
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Set pad_token as eos_token to handle padding
tokenizer.pad_token = tokenizer.eos_token

# Load and tokenize dataset
dataset = load_dataset("text", data_files={"train": "capital_cities.txt"})

def tokenize_function(examples):
    # Tokenize the text with padding and truncation
    tokenized_output = tokenizer(
        examples["text"], 
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    
    # Duplicate input_ids to labels and set padding token to -100 for labels
    tokenized_output["labels"] = tokenized_output["input_ids"].clone()
    tokenized_output["labels"][tokenized_output["labels"] == tokenizer.pad_token_id] = -100
    
    return tokenized_output

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./gpt2-capitals",
    overwrite_output_dir=True,
    num_train_epochs=3,           # Adjust as needed
    per_device_train_batch_size=1, # Adjust based on your GPU memory
    save_steps=500,
    save_total_limit=2,
    logging_dir="./logs",
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
)

# Fine-tune the model
trainer.train()

# Save the model and tokenizer
trainer.save_model()  # Saves the model weights
tokenizer.save_pretrained("./gpt2-capitals")  # Saves the tokenizer files
