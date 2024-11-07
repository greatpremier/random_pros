from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load fine-tuned model and tokenizer
model = GPT2LMHeadModel.from_pretrained("./gpt2-capitals")
tokenizer = GPT2Tokenizer.from_pretrained("./gpt2-capitals")

# Test the fine-tuned model
input_text = "What is the largest city in the world"
input_ids = tokenizer.encode(input_text, return_tensors="pt")
output = model.generate(input_ids, max_length=50)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
