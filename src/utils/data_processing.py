from datasets import load_dataset, disable_caching

def load_raw_data():
    # Disable caching to avoid issues
    disable_caching()

    # Load raw dataset (e.g., CodeSearchNet)
    dataset = load_dataset("code_search_net", "python")
    print(f"Raw dataset loaded. Size: {len(dataset['train'])} samples.")
    return dataset

def preprocess_data(dataset):
    # Preprocess the dataset (e.g., tokenize code)
    tokenized_dataset = dataset.map(lambda x: {"code": x["func_code_string"].lower()})
    print(f"Dataset preprocessed. Size: {len(tokenized_dataset['train'])} samples.")
    return tokenized_dataset

def save_processed_data(dataset, path):
    # Save processed dataset to a file
    dataset.save_to_disk(path)
    print(f"Processed dataset saved to {path}.")

if __name__ == "__main__":
    # Load and preprocess data
    raw_data = load_raw_data()
    processed_data = preprocess_data(raw_data)
    save_processed_data(processed_data, "data/processed/processed_dataset")