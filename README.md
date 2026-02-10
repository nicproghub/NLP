# NLP Text Classification Pipeline

## 📌 Project Overview

This project implements an **end-to-end Natural Language Processing (NLP) text classification pipeline** that compares a **classical machine learning baseline** against a **modern Transformer-based model** on a real-world sentiment analysis task (IMDb movie reviews).

The goal is to evaluate the trade-offs between **speed, interpretability, and performance** when using traditional feature-based models versus deep learning approaches.

---

## 🎯 Objectives

- Build a strong **classical NLP baseline** using TF-IDF and Logistic Regression  
- Fine-tune a **pretrained Transformer (DistilBERT)** for sentiment classification  
- Compare models using standard classification metrics  
- Demonstrate a **reproducible NLP workflow** from preprocessing to evaluation  

---

## 🧠 Models Compared

### 1. Classical Machine Learning Baseline
**Pipeline:**
- Text cleaning and normalization with **NLTK**
- Feature extraction using **TF-IDF**
- Classification using **Logistic Regression**

**Advantages:**
- Fast training and inference
- Highly interpretable feature weights
- Strong baseline for many text tasks

---

### 2. Transformer-Based Model
**Pipeline:**
- Pretrained **DistilBERT**
- Fine-tuned using **Hugging Face Transformers**
- Trained with the `Trainer` API (GPU-accelerated in Google Colab)

**Advantages:**
- Context-aware word representations
- Superior performance on nuanced language
- Minimal feature engineering required

---

## 📂 Dataset

- **IMDb Movie Reviews Dataset**
- Binary sentiment labels: **positive / negative**
- Split into:
  - Training set
  - Validation set
  - Held-out test set

---

## 🔄 Pipeline Workflow

1. **Data Preprocessing**
   - Text cleaning (lowercasing, punctuation removal)
   - Tokenization and stopword handling (NLTK)

2. **Feature Extraction**
   - TF-IDF vectorization for classical ML
   - Subword tokenization for DistilBERT

3. **Model Training**
   - Logistic Regression on TF-IDF features
   - Fine-tuning DistilBERT using Hugging Face Trainer

4. **Evaluation**
   - Accuracy
   - Precision
   - Recall
   - F1-score  
   Evaluated on both validation and test sets

5. **Interpretability**
   - Inspection of influential TF-IDF features
   - Qualitative comparison of model behavior

---

## 📊 Results & Evaluation

- Both models are evaluated using the same metrics for fair comparison
- Classical baseline provides strong performance with high interpretability
- DistilBERT achieves improved performance by capturing contextual semantics
- Results highlight the **performance vs. complexity trade-off** in NLP systems

---

## 🛠 Tools & Libraries

- **Python**
- **NLTK** – text preprocessing
- **scikit-learn** – TF-IDF, Logistic Regression, evaluation metrics
- **Hugging Face Transformers** – DistilBERT fine-tuning
- **PyTorch**
- **Google Colab** – GPU training environment

---

