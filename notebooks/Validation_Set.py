from transformers import AutoModelForTokenClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, load_metric
import numpy as np
import os

# Load the seqeval metric
metric = load_metric("seqeval")

# Define label names
label_names = ["O", "B-Product", "I-Product", "B-LOC", "I-LOC", "B-PRICE", "I-PRICE"]

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    pred_labels = [[label_names[p] for p, l in zip(pred, label) if l != -100] for pred, label in zip(predictions, labels)]
    results = metric.compute(predictions=pred_labels, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# Load validation dataset (adjust based on your CoNLL file or dataset)
# Example: Loading from a CoNLL file
def load_conll_file(file_path):
    # Custom function to parse CoNLL format
    sentences, labels = [], []
    current_sentence, current_labels = [], []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                token, label = line.split()
                current_sentence.append(token)
                current_labels.append(label)
            else:
                if current_sentence:
                    sentences.append(current_sentence)
                    labels.append(current_labels)
                    current_sentence, current_labels = [], []
    return {"tokens": sentences, "ner_tags": labels}

# Replace with your validation file path
validation_data = load_conll_file("path_to_validation_conll.txt")
validation_dataset = Dataset.from_dict(validation_data)

# Tokenize validation dataset
def tokenize_and_align_labels(examples, tokenizer):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True, padding=True)
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        aligned_labels = [-100 if word_id is None else label_names.index(label[word_id]) for word_id in word_ids]
        labels.append(aligned_labels)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# List of fine-tuned model paths
models = [
    "./results/xlmr_model",  # Replace with actual path
    "./results/distilbert_model",
    "./results/mbert_model"
]

# Training arguments for evaluation
training_args = TrainingArguments(
    output_dir="./eval_results",
    evaluation_strategy="epoch",
    per_device_eval_batch_size=16,
)

# Evaluate each model
for model_path in models:
    if not os.path.exists(model_path):
        print(f"Model path {model_path} does not exist. Skipping...")
        continue
    print(f"Evaluating model: {model_path}")
    
    # Load model and tokenizer
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Tokenize validation dataset
    tokenized_validation = validation_dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer), batched=True
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        eval_dataset=tokenized_validation,
    )
    
    # Evaluate
    results = trainer.evaluate()
    print(f"Results for {model_path}:")
    print(results)