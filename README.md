# 🧠 Verdict AI – Human vs AI Sentence Classifier

Verdict is a lightweight machine learning engine that distinguishes between **human-written** and **AI-generated** sentences using natural language features and a trained classifier.

## 🔍 What It Does

From a single sentence, Verdict analyzes linguistic structure, statistical complexity, and stylistic patterns to determine:  
**Was this written by a human—or a large language model?**

## 🚀 Project Structure

```verdict-ai/
├── app/
│ ├── sentence_dataset_full.csv # Full labeled dataset (10,000+ rows)
│ ├── feature_dataset_v2.csv # Feature-extracted dataset
│ ├── verdict_model.pkl # Trained classifier model
│ ├── features.py # Custom feature extraction logic
│ ├── generate_features.py # Converts sentences → features
│ ├── train_model.py # Trains + evaluates the model
│ └── test_features.py # Manual test scripts
├── requirements.txt
├── .gitignore
└── README.md
```

## 🧩 How It Works

### 1. Load Labeled Sentences  
Each row in `sentence_dataset_full.csv` contains:
- A single sentence
- A label: `1 = Human`, `0 = AI`

### 2. Extract Natural Language Features  
The engine pulls out:
- 🧠 `avg_sentence_length`: Average words per sentence  
- 🌀 `sentence_variance`: Sentence-to-sentence complexity  
- 📚 `lexical_diversity`: Unique/total word ratio  
- 📖 `readability_score`: Flesch score  
- 📈 `entropy`: Character-level entropy (distributional complexity)  
- 🔢 `word_count`: Total words

### 3. Train a Classifier  
- Model: `RandomForestClassifier` (scikit-learn)
- Data: **10,076 real sentences** (balanced: 5,038 human / 5,038 AI)
- Split: 80/20 stratified train/test
- Result:  
  - ✅ **Accuracy:** 81%  
  - ✅ Precision/Recall/F1 all balanced at 0.81  
  - ✅ Confusion matrix shows stable generalization

### 4. Save the Model  
- Saved as `app/verdict_model.pkl` using `joblib`
- Easily loaded for future prediction tools

## 🧪 Results (as of June 18, 2025)

**Test Set Performance on 2,016 Sentences**
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| AI    | 0.81      | 0.80   | 0.81     | 1008    |
| Human | 0.81      | 0.81   | 0.81     | 1008    |

**Overall Accuracy:** **0.81**  
**Confusion Matrix:**
[[811 197]
[190 818]]


---

## 📦 Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/neuron-cloud/verdict-ai.git
cd verdict-ai

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train or retrain the model
python3 app/train_model.py
🔮 Coming Soon
predict_sentence.py – CLI or Colab tool to test new sentences

Web interface for live, user-facing verdicts

Model explainability dashboard (SHAP/LIME)

Ensemble + LLM-enhanced v2 architecture

🧠 Author
Built by Mychael Delgardo, Columbia-trained physician-scientist turned builder.
Verdict is part of a broader AI portfolio focused on clinical reasoning, recovery forecasting, and cognitive detection.

Let’s build smarter tools that think with us—not for us.
