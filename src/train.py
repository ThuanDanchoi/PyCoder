import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_from_disk, DatasetDict

def load_pretrained_model():
    # Load pretrained model and tokenizer
    model_name = "Salesforce/codegen-350M-mono"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Thêm padding token cho tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    
    return model, tokenizer

def load_processed_data():
    # Load processed dataset
    dataset = load_from_disk("data/processed/processed_dataset")
    
    # If the dataset is not split, create a DatasetDict with a "train" key
    if not isinstance(dataset, DatasetDict):
        dataset = DatasetDict({"train": dataset})
    
    print(f"Processed dataset loaded. Size: {len(dataset['train'])} samples.")
    print(f"Sample data: {dataset['train'][0]}")  # Print a sample to verify the structure
    print(f"Dataset features: {dataset['train'].features}")  # Print dataset features
    return dataset

def preprocess_dataset(dataset, tokenizer):
    # Hàm xử lý từng batch dữ liệu
    def preprocess_function(examples):
        # Tokenize code
        model_inputs = tokenizer(
            examples["func_code_string"],
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors=None  # Thay đổi từ "pt" thành None
        )
        
        # Tạo labels giống với input_ids cho mô hình causal LM
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        
        return model_inputs

    # Áp dụng preprocessing cho toàn bộ dataset
    processed_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    
    return processed_dataset

def fine_tune_model(model, tokenizer, dataset):
    # Xử lý dữ liệu trước khi huấn luyện
    processed_dataset = preprocess_dataset(dataset, tokenizer)
    
    # Thiết lập training arguments
    training_args = TrainingArguments(
        output_dir="models/fine_tuned/",
        per_device_train_batch_size=4,
        num_train_epochs=3,
        save_steps=500,
        save_total_limit=2,
        fp16=False,
        logging_dir="outputs/logs/",
    )

    # Khởi tạo trainer với dataset đã xử lý
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset["train"],
    )

    print("Starting training...")
    trainer.train()
    print("Training completed.")

if __name__ == "__main__":
    # Load pretrained model and data
    model, tokenizer = load_pretrained_model()
    dataset = load_processed_data()

    # Fine-tune the model
    fine_tune_model(model, tokenizer, dataset)