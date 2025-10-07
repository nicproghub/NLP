# NLP Text Classification Pipeline

End-to-end NLP project that compares a **classical ML baseline (TF-IDF + Logistic Regression)** against a **fine-tuned Transformer (DistilBERT)** on real-world text classification (IMDb sentiment). The pipeline covers **preprocessing (NLTK)**, **feature extraction**, **training & evaluation**, basic **interpretability**, and **reproducible outputs**.

---

## 🔍 Overview
- **Classical baseline:** NLTK cleaning → TF-IDF → Logistic Regression (fast, interpretable).
- **Transformer model:** DistilBERT fine-tuning with Hugging Face `Trainer` (GPU-accelerated in Colab).
- **Metrics:** Accuracy, Precision, Recall, F1 on **validation** and **held-out test** sets.


