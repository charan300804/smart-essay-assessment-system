# 🎓 Smart Essay Assessment and Scoring System (SEAS)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

An intelligent, AI-powered web application designed to automatically grade essays and provide comprehensive feedback on writing quality, grammatical structure, and readability metrics. 

By combining **Deep Learning (Bi-LSTM + Attention)** with **Advanced Natural Language Processing (NLP)**, SEAS evaluates essays not just on simple word statistics, but on semantic coherence and writing style.

---

## 🌟 Key Features

*   🧠 **Deep Learning Scoring Model**: Utilizes a Bidirectional Long Short-Term Memory (Bi-LSTM) network enhanced by an Attention Mechanism to capture sequential relationships and highlight key sentences.
*   📊 **Linguistic Analytics**:
    *   **Readability Metrics**: Evaluates readability using the Flesch-Kincaid grade level index.
    *   **Vocabulary & POS Distribution**: Visualizes parts of speech (nouns, verbs, adjectives, etc.) distribution using interactive charts.
    *   **Word Cloud**: Dynamically generates word clouds highlighting the most prominent terms in the essay.
*   🏆 **Gamification & Badging**: Gamifies writing with achievements such as "Top Scorer", "Scholar", or "Lexical Genius" based on vocabulary richness and essay length.
*   💾 **Persistent Database**: Automatically saves submitted essays, readability scores, and model evaluations in a local SQLite database for history tracking.
*   📁 **File Uploader**: Paste essays directly or upload plain `.txt` files for instant scoring.
*   🎨 **Glassmorphic UI**: Beautiful frontend built with Streamlit featuring a clean, responsive layout, dark-mode elements, and metric gauges.

---

## 🛠️ Technology Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Frontend Dashboard** | Streamlit | Rich, interactive dashboard with custom glassmorphic styling. |
| **Backend API** | FastAPI / Uvicorn | Asynchronous web framework exposing model prediction endpoints. |
| **Deep Learning** | Keras / TensorFlow | Bi-LSTM neural network with attention layer for essay scoring. |
| **NLP Pipeline** | SpaCy & NLTK | Tokenization, POS tagging, readability analysis, and text cleaning. |
| **Database** | SQLite | Serverless database engine for persisting evaluation history. |

---

## 📁 Repository Structure

```
project/
├── data/                  # [Ignored] Datasets and GloVe embeddings
│   ├── training_set_rel3.tsv
│   └── glove.6B.100d.txt
├── models/                # [Ignored] Saved models and tokenizers
│   ├── final_model.h5
│   └── tokenizer.pickle
├── src/                   # Source files
│   ├── api.py             # FastAPI backend server
│   ├── app.py             # Streamlit frontend application
│   ├── config.py          # Project configuration values
│   ├── data_loader.py     # Utilities for loading the ASAP dataset
│   ├── database.py        # SQLite history tracking setup
│   ├── model.py           # Bi-LSTM + Attention neural network architecture
│   ├── preprocess.py      # Feature engineering and NLP preprocessor
│   └── train.py           # Model training and evaluation script
├── .gitignore             # Configured git ignore file
├── docker-compose.yml     # Multi-container orchestration config
├── Dockerfile             # Container configuration for app deployment
├── requirements.txt       # Python package dependencies
└── run_locally.bat        # Windows batch script for one-click startup
```

---

## 🚀 Setup & Installation Guide

### Prerequisites
*   **Python 3.8 - 3.11** installed.
*   **Git** (to clone and push repository).

---

### Step 1: Clone and Environment Setup

1. Clone this repository and navigate to the project directory:
   ```bash
   git clone https://github.com/charan300804/smart-essay-assessment-system.git
   cd smart-essay-assessment-system
   ```

2. Create a virtual environment and activate it:
   ```bash
   # Windows (Command Prompt)
   python -m venv venv
   call venv\Scripts\activate

   # macOS / Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install all required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the SpaCy English language model:
   ```bash
   python -m spacy download en_core_web_sm
   ```

---

### Step 2: Data Preparation 📂

The model relies on GloVe word representations and the ASAP dataset for training. These files are excluded from the repository due to their size and must be downloaded manually:

1. Create a `data/` directory in the root of the project.
2. Download the **ASAP-AES** dataset from Kaggle:
   *   Visit [Kaggle ASAP-AES Data](https://www.kaggle.com/c/asap-aes/data).
   *   Download `training_set_rel3.tsv` and place it in the `data/` folder.
3. Download the pre-trained **GloVe Embeddings**:
   *   Visit [Stanford GloVe NLP Page](https://nlp.stanford.edu/projects/glove/).
   *   Download `glove.6B.zip` (contains 100d, 200d, etc. vectors).
   *   Extract `glove.6B.100d.txt` and place it in the `data/` folder.

---

### Step 3: Train the Deep Learning Model 🧠

Train the Bi-LSTM model with Attention locally using the training script:

```bash
python -m src.train
```

*   The script runs for 50 epochs and automatically saves the best performing model based on validation metrics.
*   Once training completes, it produces `models/final_model.h5` and `models/tokenizer.pickle`.

---

### Step 4: Run the Application 💻

You can start both the backend FastAPI service and the Streamlit frontend.

#### Option A: One-Click Startup (Windows)
Double-click the **`run_locally.bat`** file in the root folder. It will launch two command windows: one running the FastAPI server and another running Streamlit.

#### Option B: Manual Execution (Two Terminals)

1. **Terminal 1: Start Backend API**
   ```bash
   uvicorn src.api:app --host 0.0.0.0 --port 8000
   ```
   *Verify the API is active by visiting `http://localhost:8000/docs` in your browser.*

2. **Terminal 2: Start Streamlit Frontend**
   ```bash
   streamlit run src/app.py
   ```
   *The dashboard will automatically open at `http://localhost:8501`.*

---

## 🎨 Architecture & Deep Learning Details

The neural network is structured to evaluate coherence, vocabulary usage, and grammatical structure:
1. **Embedding Layer**: Preloaded with **GloVe (100d)** word embeddings to map words to their semantic vectors.
2. **Bidirectional LSTM Layer**: Processes sequences forward and backward, allowing the model to capture sentence flow and contextual relations.
3. **Attention Layer**: Learns to weight parts of the essay that have a higher impact on writing quality.
4. **Dense Feedforward Layer**: Maps output dimensions to predict the score on a scale of `0.0` to `10.0`.

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
