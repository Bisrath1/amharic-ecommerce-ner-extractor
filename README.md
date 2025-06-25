# Amharic E-commerce Named Entity Recognition (NER) System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![NLP](https://img.shields.io/badge/NLP-Amharic-green)

A transformer-based NER system to extract products, prices, and locations from Ethiopian Telegram e-commerce channels for EthioMart's centralized platform.

## 🚀 Project Overview
**Business Need**: EthioMart aims to consolidate decentralized Telegram e-commerce channels by:
- Extracting key entities (products/prices/locations) from Amharic text
- Creating vendor scorecards for micro-lending eligibility
- Building a unified e-commerce database

**Key Features**:
- Telegram data scraper for Ethiopian channels
- Rule-based + fine-tuned LLM (XLM-Roberta/mBERT) NER pipeline
- Vendor analytics engine with lending score calculation
- Model interpretability using SHAP/LIME


## 🔧 Installation
1. Clone repository:
   ```bash
   git clone https://github.com/Bisrath1/amharic-ecommerce-ner-extractor.git
   cd amharic-ecommerce-ner-extractor
