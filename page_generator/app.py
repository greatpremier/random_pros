from transformers import T5Tokenizer, T5ForConditionalGeneration

# Define the path to the model directory
model_path = './results'

# Check if the directory exists and contains the necessary files
'''import os
if os.path.exists(model_path) and all(os.path.isfile(os.path.join(model_path, file)) for file in [
    'config.json', 'pytorch_model.bin', 'special_tokens_map.json', 'tokenizer_config.json', 'vocab.json']):
    print("Model directory and files found.")
else:
    print("Model directory or files missing.")

# Load the tokenizer and model
try:
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)'''


tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

# Prepare the input prompt
input_prompt = "Generate HTML for a webpage with a title and a paragraph"
input_ids = tokenizer.encode(input_prompt, return_tensors='pt')

# Generate HTML
outputs = model.generate(
    input_ids, 
    max_length=512, 
    num_beams=5, 
    early_stopping=True, 
    no_repeat_ngram_size=2, 
    temperature=0.7
)

# Decode the output
generated_html = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(generated_html)
