import time
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

def measure_inference_time(model, tokenizer, inputs, device="cuda" if torch.cuda.is_available() else "cpu"):
    model.to(device)
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        for input_text in inputs:
            tokenized_input = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True).to(device)
            model(**tokenized_input)
    end_time = time.time()
    return (end_time - start_time) / len(inputs)  # Average time per input

# Define model paths (replace with your actual fine-tuned model paths)
models = [
    "./results/xlmr_model",  # Path to fine-tuned XLM-RoBERTa model
    "./results/distilbert_model",  # Path to fine-tuned DistilBERT model
    "./results/mbert_model"  # Path to fine-tuned mBERT model
]

# Example inputs (list of Amharic messages)
inputs = [
    "የልጆች ጫማ በ 500 ብር በ አዲስ አበባ",
    "ስልክ በ 12000 ብር በ ቦሌ",
    "ቲሸርት በ 300 ብር በ መገናኛ",
    "ላፕቶፕ በ 25000 ብር በ ፒያሳ",
    "የእንግዳ መኝታ በ 1500 ብር በ ሜክሲኮ"
]

# Evaluate inference time for each model
for model_path in models:
    try:
        # Load model and tokenizer
        model = AutoModelForTokenClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        avg_time = measure_inference_time(model, tokenizer, inputs, device="cuda" if torch.cuda.is_available() else "cpu")
        print(f"Average inference time for {model_path}: {avg_time:.4f} seconds")
    except Exception as e:
        print(f"Error loading or evaluating {model_path}: {str(e)}")