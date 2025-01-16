from transformers import AutoModelForCausalLM, AutoTokenizer

def load_fine_tuned_model():
    # Load fine-tuned model and tokenizer
    model_name = "models/fine_tuned/"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def generate_code(model, tokenizer, prompt):
    # Generate code from a prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    # Load model and generate code
    model, tokenizer = load_fine_tuned_model()
    prompt = "Write a Python function to calculate the factorial of a number:"
    generated_code = generate_code(model, tokenizer, prompt)

    # Save results
    with open("outputs/results/generated_code.txt", "w") as f:
        f.write(generated_code)

    print("Generated code saved to outputs/results/generated_code.txt")