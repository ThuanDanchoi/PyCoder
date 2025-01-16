import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load fine-tuned model and tokenizer
model_name = "models/fine_tuned/"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Streamlit app
st.title("Python Code Generator")
code_input = st.text_area("Enter your Python code prompt:")

if st.button("Generate Code"):
    if code_input:
        # Generate code
        inputs = tokenizer(code_input, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=100)
        generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Display generated code
        st.code(generated_code)
    else:
        st.warning("Please enter a code prompt.")