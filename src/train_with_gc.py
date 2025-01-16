import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_from_disk, DatasetDict
from google.colab import drive
import psutil
import os

def mount_drive():
    # Mount Google Drive
    drive.mount('/content/drive')
    # Tạo thư mục project trong drive nếu chưa có
    project_path = "/content/drive/MyDrive/PyCoder"
    os.makedirs(project_path, exist_ok=True)
    return project_path

def load_pretrained_model():
    # Kiểm tra và sử dụng GPU nếu có
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model_name = "Salesforce/codegen-350M-mono"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    
    # Chuyển model lên GPU
    model = model.to(device)
    return model, tokenizer

def load_processed_data(project_path):
    dataset_path = os.path.join(project_path, "data/processed/processed_dataset")
    dataset = load_from_disk(dataset_path)
    
    if not isinstance(dataset, DatasetDict):
        dataset = DatasetDict({"train": dataset})
    
    print(f"Processed dataset loaded. Size: {len(dataset['train'])} samples")
    return dataset

def fine_tune_model(model, tokenizer, dataset, project_path):
    processed_dataset = preprocess_dataset(dataset, tokenizer)
    
    # Tối ưu hóa cho Google Colab
    training_args = TrainingArguments(
        output_dir=f"{project_path}/models/fine_tuned/",
        per_device_train_batch_size=8,  # Tăng batch size vì có GPU
        num_train_epochs=3,
        save_steps=1000,
        save_total_limit=2,
        fp16=True,  # Sử dụng mixed precision training
        gradient_accumulation_steps=4,
        logging_dir=f"{project_path}/outputs/logs/",
        logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        report_to="tensorboard",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset["train"],
    )

    # Theo dõi tài nguyên
    def log_gpu_memory():
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**2
            cached = torch.cuda.memory_reserved() / 1024**2
            print(f"GPU Memory: Allocated: {allocated:.2f}MB, Cached: {cached:.2f}MB")
    
    print("Starting training...")
    log_gpu_memory()
    trainer.train()
    print("Training completed.")
    
    # Lưu model về Google Drive
    model_save_path = f"{project_path}/models/fine_tuned/final_model"
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)

if __name__ == "__main__":
    # Mount Google Drive và setup paths
    project_path = mount_drive()
    
    # Load model và data
    model, tokenizer = load_pretrained_model()
    dataset = load_processed_data(project_path)
    
    # Huấn luyện model
    fine_tune_model(model, tokenizer, dataset, project_path)