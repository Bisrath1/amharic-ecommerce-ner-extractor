# Amharic E-commerce Data Extractor Named Entity Recognition (NER) System

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![NLP](https://img.shields.io/badge/NLP-Amharic-green)

A transformer-based NER system to extract products, prices, and locations from Ethiopian Telegram e-commerce channels for EthioMart's centralized platform.


## Project Overview
This project, developed as part of the **10 Academy Artificial Intelligence Mastery** (18 June - 24 June 2025), builds an **Amharic E-commerce Data Extractor** to transform unstructured Telegram posts into a structured FinTech engine for EthioMart. The system extracts key entities (Product, Price, Location) from Amharic text in Ethiopian e-commerce Telegram channels, enabling a centralized platform for product discovery and vendor analysis for micro-lending.

## Business Need
EthioMart aims to consolidate decentralized Telegram-based e-commerce activities in Ethiopia into a unified platform. By extracting structured data from Telegram posts, the system enables:
- Seamless customer interaction with multiple vendors.
- A centralized database of products, prices, and locations.
- A FinTech engine to identify promising vendors for micro-lending based on engagement metrics.

## Objectives
1. Develop a repeatable workflow for data ingestion, preprocessing, and entity extraction.
2. Fine-tune a transformer-based model for Amharic NER with high F1-score accuracy.
3. Compare multiple NER models and recommend the best for EthioMart’s use case.
4. Use SHAP and LIME for model interpretability to ensure transparency.
5. Create a vendor scorecard to rank vendors for micro-lending based on activity and engagement.

## Repository Structure
```
Amharic-Ecommerce-Extractor/
├── data/
│   ├── raw/                    # Raw Telegram data (CSV/JSON)
│   ├── processed/              # Preprocessed data and CoNLL files
│   └── vendor_scorecard.csv    # Final vendor scorecard
├── notebooks/
│   ├── task1_data_ingestion.ipynb        # Data scraping and preprocessing
│   ├── task2_data_labeling.ipynb         # CoNLL format labeling
│   ├── task3_ner_finetuning.ipynb        # NER model fine-tuning
│   ├── task4_model_comparison.ipynb      # Model comparison and selection
│   ├── task5_model_interpretability.ipynb # SHAP and LIME analysis
│   └── task6_vendor_scorecard.ipynb      # Vendor analytics engine
├── models/
│   └── finetuned_model/        # Saved fine-tuned NER model
├── reports/
│   ├── interpretability_report.txt # Model interpretability findings
│   └── vendor_scorecard_report.txt # Vendor scorecard summary
├── src/
│   ├── data_ingestion.py       # Telegram data scraping scripts
│   ├── preprocessing.py         # Text preprocessing utilities
│   ├── ner_utils.py            # NER model processing functions
│   └── vendor_analytics.py     # Vendor scorecard calculations
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Installation
1. **Clone the Repository**:
```
git clone https://github.com/your-repo/Amharic-Ecommerce-Extractor.git
cd Amharic-Ecommerce-Extractor
```

2. **Set Up Environment**:
- Use Python 3.8+.
- Install dependencies:
```
pip install -r requirements.txt
```
- Key libraries: `transformers`, `datasets`, `pandas`, `numpy`, `shap`, `lime`, `torch`.

3. **Optional: GPU Setup**:
- For faster training, use Google Colab or a local machine with CUDA-enabled GPU.
- Install `torch` with CUDA support if needed.

## Usage
The project is divided into six tasks, each implemented in a Jupyter notebook or Python script. Follow these steps to run the pipeline:

### Task 1: Data Ingestion and Preprocessing
- **Script**: `notebooks/task1_data_ingestion.ipynb`
- **Description**: Scrapes messages from at least 5 Ethiopian e-commerce Telegram channels, preprocesses Amharic text (tokenization, normalization), and stores data in `data/processed/`.
- **Run**:
```
jupyter notebook notebooks/task1_data_ingestion.ipynb
```
- **Output**: CSV/JSON files with columns: `vendor_id`, `message`, `views`, `timestamp`.

### Task 2: Label a Subset of Dataset in CoNLL Format
- **Script**: `notebooks/task2_data_labeling.ipynb`
- **Description**: Labels 30–50 messages in CoNLL format for entities (Product, Price, Location) using the `Message` column from the dataset.
- **Entity Types**: `B-Product`, `I-Product`, `B-LOC`, `I-LOC`, `B-PRICE`, `I-PRICE`, `O`.
- **Run**:
```
jupyter notebook notebooks/task2_data_labeling.ipynb
```
- **Output**: `data/processed/labeled_data.conll`

### Task 3: Fine-Tune NER Model
- **Script**: `notebooks/task3_ner_finetuning.ipynb`
- **Description**: Fine-tunes a transformer model (e.g., XLM-RoBERTa, bert-tiny-amharic, or afroxmlr) on the labeled CoNLL dataset for Amharic NER.
- **Run**:
```
jupyter notebook notebooks/task3_ner_finetuning.ipynb
```
- **Output**: Fine-tuned model saved in `models/finetuned_model/`.

### Task 4: Model Comparison & Selection
- **Script**: `notebooks/task4_model_comparison.ipynb`
- **Description**: Compares multiple NER models (e.g., XLM-RoBERTa, DistilBERT, mBERT) based on F1-score, precision, recall, and computational efficiency.
- **Run**:
```
jupyter notebook notebooks/task4_model_comparison.ipynb
```
- **Output**: Evaluation metrics and model selection recommendation.

### Task 5: Model Interpretability
- **Script**: `notebooks/task5_model_interpretability.ipynb`
- **Description**: Uses SHAP and LIME to explain NER model predictions, analyzing difficult cases and ensuring transparency.
- **Run**:
```
jupyter notebook notebooks/task5_model_interpretability.ipynb
```
- **Output**: SHAP plots (`shap_plot.png`), LIME HTML files (`lime_explanation.html`), and interpretability report (`reports/interpretability_report.txt`).

### Task 6: FinTech Vendor Scorecard for Micro-Lending
- **Script**: `notebooks/task6_vendor_scorecard.ipynb`
- **Description**: Builds a Vendor Analytics Engine to calculate metrics (Avg. Views/Post, Posts/Week, Avg. Price, Lending Score) and rank vendors for micro-lending.
- **Run**:
```
jupyter notebook notebooks/task6_vendor_scorecard.ipynb
```
- **Output**: Vendor scorecard (`data/vendor_scorecard.csv`) and report section (`reports/vendor_scorecard_report.txt`).

## Data Sources
- **Telegram Channels**: At least 5 Ethiopian e-commerce channels (e.g., Shageronlinestore).
- **Amharic NER Dataset**: Labeled dataset for training (e.g., Amharic news NER dataset).
- **Format**: Text (Amharic messages), metadata (views, timestamps), and images (not processed in this version).

## Key Metrics
- **NER Performance**: F1-score, precision, recall for entity extraction.
- **Vendor Metrics**:
  - **Posts/Week**: Average number of posts per week.
  - **Avg. Views/Post**: Average views per post.
  - **Top Post**: Product and price of the highest-viewed post.
  - **Avg. Price (ETB)**: Average product price.
  - **Lending Score**: Weighted score (e.g., 0.5 * normalized Avg. Views + 0.5 * normalized Posts/Week).

## Results
- **NER Model**: Fine-tuned model achieves high F1-score for Product, Price, and Location entities.
- **Interpretability**: SHAP and LIME reveal key token contributions and highlight areas for improvement (e.g., ambiguous terms like "Bole").
- **Vendor Scorecard**: Ranks vendors for micro-lending, identifying high-engagement and consistent vendors.


## License
This project is licensed under the MIT License.
