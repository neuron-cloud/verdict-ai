# 🧠 Verdict AI – Human vs AI Sentence Classifier

Verdict is a lightweight machine learning engine that distinguishes between human-written and AI-generated sentences using natural language features and a trained classifier.

## 🚀 Project Structure

verdict-ai/
├── app/
│ ├── train_model.py # trains and saves model
│ ├── sentence_dataset_full.csv # raw labeled sentences
│ ├── feature_dataset_v2.csv # extracted features
│ └── verdict_model.pkl # trained model (output)
├── requirements.txt # dependencies
├── .gitignore # ignored files
└── README.md


## 🧩 How It Works

### 1. Load Your Data  
- `sentence_dataset_full.csv`: Each row is a sentence labeled "HUMAN" or "AI".

### 2. Extract Features (done before training)
- Average sentence length
- Variance in word length
- Lexical diversity (unique/total words)
- Readability (Flesch)
- Entropy (distributional complexity)
- Word count

### 3. Train a Model  
- Uses `RandomForestClassifier` from `scikit-learn`
- Balanced dataset: 50 AI vs 50 HUMAN samples
- Train/test split: 80/20 with stratification
- Accuracy: **85%**

### 4. Save & Reuse  
- The trained model is saved to `app/verdict_model.pkl` using `joblib`
- Future prediction scripts can load this and classify new sentences

## 📦 Getting Started

bash
# 1. Clone the repo
git clone https://github.com/neuron-cloud/verdict-ai.git
cd verdict-ai

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model
python3 app/train_model.py

## 🔮 Coming Soon
predict_sentence.py – CLI or Colab tool to test new sentences

Web interface for live verdicts

Model explainability dashboard

## 🧠 Author
Built by Mychael Delgardo as part of a personal AI/ML portfolio.
Let’s build smarter tools that think with us, not for us.
