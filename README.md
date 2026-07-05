# 📰 Fake News Detection AI

A machine learning web app that classifies news articles as **Real** or **Fake** using NLP text preprocessing, TF-IDF vectorization, and a Logistic Regression classifier — wrapped in an interactive Streamlit UI.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.39-red)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-1.5-orange)
![Accuracy](https://img.shields.io/badge/Accuracy-97%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

🔗 **Live Demo:** _add your Streamlit Community Cloud link here after deploying_
🔗 **Portfolio:** [nadir789259.github.io](https://nadir789259.github.io)

---

## 🎯 Features

- ✅ Classifies any pasted news article as **Real** or **Fake** in real time
- 📊 Shows confidence score and a full fake/real probability breakdown
- 🧠 NLP pipeline: lowercasing → punctuation removal → tokenization → stopword removal → Porter stemming
- ⚡ Model and vectorizer cached with `st.cache_resource` for fast repeat predictions
- 🎨 Custom dark-themed UI via external CSS
- 🛡️ Input validation and exception handling (empty input, short text, missing model files)

## 🖥️ Screenshots

_Add 1–2 screenshots or a short GIF of the app here (drag & drop into this section on GitHub, or place images in a `screenshots/` folder and reference them, e.g. `![App Screenshot](screenshots/demo.png)`)._

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.11 |
| Web App | Streamlit |
| ML | Scikit-learn (Logistic Regression, TF-IDF) |
| NLP | NLTK (stopwords, tokenization, Porter Stemmer) |
| Data | Pandas, NumPy |

## 📁 Project Structure

```
Fake-News-Detection-AI/
│
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── LICENSE                   # MIT License
├── style.css                 # Custom UI styling
├── fake_news_model.pkl       # Trained Logistic Regression model
├── tfidf_vectorizer.pkl      # Fitted TF-IDF vectorizer
├── Notebook/
│   └── Fake_news_Detector.ipynb   # EDA, preprocessing & model training
├── Dataset/
│   └── news.csv               # Training data (see Dataset section below)
└── README.md
```

## 📊 Dataset

This project is trained on a cleaned subset of the **WELFake** dataset (~45,757 labeled articles, a near-even split of real and fake news), originally built by merging four public news datasets (Kaggle, McIntire, Reuters, BuzzFeed Political) — [source on Kaggle](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification).

> **Note:** `Dataset/news.csv` (~120 MB) is excluded from this repo via `.gitignore` because it exceeds GitHub's 100 MB per-file limit. Download it from the Kaggle link above and place it in `Dataset/news.csv` if you want to re-run the notebook.

## 📈 Model Performance

Evaluated on a held-out 20% test split (9,152 articles):

| Metric | Score |
|---|---|
| Accuracy | 97.02% |
| Precision | 97.17% |
| Recall | 96.86% |
| F1 Score | 97.01% |

**Confusion Matrix**

| | Predicted Fake | Predicted Real |
|---|---|---|
| **Actual Fake** | 4,443 | 129 |
| **Actual Real** | 144 | 4,436 |

## ⚙️ How It Works

1. Raw article text is cleaned: lowercased, punctuation stripped, tokenized, stopwords removed, and words reduced to their root form via Porter Stemming.
2. The cleaned text is transformed into numerical features using the **same fitted TF-IDF vectorizer** used during training.
3. A **Logistic Regression** model predicts the class (Real/Fake) and returns class probabilities.
4. The app displays the prediction, confidence score, and a probability breakdown.

> ⚠️ The exact same cleaning steps used to train the model in the notebook are used at prediction time in `app.py`. This matters: if the two pipelines drift apart, the words the model sees at prediction time won't match what it learned from during training, which quietly degrades accuracy.

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/NADIR789259/Fake-News-Detection-AI.git
cd Fake-News-Detection-AI

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`.

## ⚠️ Disclaimer

This tool provides an AI-based prediction from patterns learned in training data. It is a portfolio/educational project and should not be relied on as the sole method for verifying news authenticity — always cross-check with trusted sources.

## 🔮 Future Improvements

- [ ] Deploy live on Streamlit Community Cloud and link it above
- [ ] Add a source-credibility signal (domain reputation) alongside the text model
- [ ] Experiment with a Random Forest / XGBoost comparison, similar to prior model-comparison work
- [ ] Add unit tests for `clean_text()` and `predict_news()`

## 👨‍💻 Author

**Nadir Khan**
Data Analyst | Aspiring ML Engineer
🌐 [Portfolio](https://nadir789259.github.io) · 💻 [GitHub](https://github.com/NADIR789259)

## 📄 License

This project is licensed under the [MIT License](LICENSE).
