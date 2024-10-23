from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Define the path to the model directory
model_path = './results'

# Load the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

# Prepare the input prompt
input_prompt = "Generate html page with title and paragraph"
input_ids = tokenizer.encode(input_prompt, return_tensors='pt')

# Create attention mask to avoid confusion between pad token and eos token
attention_mask = input_ids.ne(tokenizer.pad_token_id)

# Generate HTML
outputs = model.generate(
    input_ids, 
    attention_mask=attention_mask,  # Provide attention mask
    max_length=512, 
    num_beams=5, 
    early_stopping=True, 
    no_repeat_ngram_size=2, 
    temperature=0.7,
    pad_token_id=tokenizer.eos_token_id  # Use eos token as pad token
)

# Decode the output
generated_html = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(generated_html)
