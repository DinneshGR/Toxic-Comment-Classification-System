# 🧪 Multi-Label Toxicity Detection Model

A deep learning-based NLP project that detects multiple types of toxic comments using the Jigsaw dataset. This model classifies text into six toxicity categories and provides real-time predictions through an interactive web interface.

---

## 📌 Overview

This project builds a **multi-label text classification system** using TensorFlow to identify harmful content in user-generated text. It processes large-scale textual data and outputs probability scores for different toxicity categories.

---

## 🚀 Key Features

- 🔍 Multi-label classification for **6 toxicity types**
- 🧠 Deep learning model using **Bidirectional LSTM**
- ⚡ Efficient TensorFlow data pipeline (caching, shuffling, prefetching)
- 🌐 Interactive **Gradio web app** for real-time predictions
- 📊 Handles large-scale vocabulary (~200K words)

---

## 🏗️ Model Architecture

### 🔤 Text Processing
- `TextVectorization` layer
- Vocabulary size: ~200,000
- Sequence length: 1800 tokens

### 🧠 Neural Network
- Bidirectional LSTM (32 units)
- Dense layers:
  - 128 → 256 → 128 neurons
- Output layer:
  - 6 units (multi-label, sigmoid activation)

---

## 📊 Dataset

- Source: Jigsaw Toxic Comment Dataset  
- Data split:
  - 70% Training
  - 20% Validation
  - 10% Testing

---

## 📈 Performance Metrics

| Metric     | Score   |
|------------|--------|
| Precision  | **83.7%** |
| Recall     | 65.6%  |
| Accuracy   | 48.7%  |

> ⚠️ Note: Accuracy is lower due to the nature of multi-label classification and class imbalance.

---

## ⚙️ Data Pipeline

Built using TensorFlow `tf.data` API:
- Caching for faster data loading
- Shuffling for improved generalization
- Prefetching for optimized performance

---

## 🌐 Deployment

- Developed using **Gradio**
- Features:
  - Real-time comment input
  - Probability scores for each toxicity class
  - Automated classification thresholds

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- NumPy / Pandas
- Gradio
- NLP preprocessing tools
