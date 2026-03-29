# Emotion & Sentiment Analyzer

A CLI-based AI tool that analyzes the **emotions** and **sentiment** hidden in any text you type — built using Python, NLP, and Machine Learning.

---

## What It Does

You type a sentence. The tool tells you:
- What **emotions** are present (happy, angry, fearful, sad, etc.)
- What **sentiment** it carries (positive, negative, neutral)
- Shows **bar graphs** for both — so you can actually see the results

No internet needed. Runs fully in the terminal.

---

## Tech Stack

| Component | Library/Tool |
|-----------|-------------|
| Language | Python 3 |
| Emotion Detection | `emotions.txt` (keyword mapping) |
| Sentiment Analysis | `scikit-learn` (Naive Bayes + CountVectorizer) |
| Graphs | `matplotlib` |
| Data | `combined_sentiment_data.csv` |
| Model Saving | `joblib` |

---

## Project Structure

```
emotion-sentiment-analyzer/
│
├── main.py                      # Main CLI application
├── emotions.txt                 # Emotion keyword dictionary
├── combined_sentiment_data.csv  # Training data for sentiment model
├── model.pkl                    # Auto-generated after first run
├── vectorizer.pkl               # Auto-generated after first run
└── README.md
```

---

## Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/emotion-sentiment-analyzer.git
cd emotion-sentiment-analyzer
```

### 2. Install dependencies

```bash
pip install matplotlib scikit-learn joblib
```

### 3. Run the program

```bash
python main.py
```

> **Note:** On first run, the ML model trains automatically from `combined_sentiment_data.csv` and saves as `model.pkl`. Subsequent runs load instantly from cache.

---

## How to Use

When you run the program, a menu appears:

```
1. Enter text and check
2. About
3. Help
4. exit
```

**Step 1** -> Select option `1`  
**Step 2** -> Type any sentence, for example:

```
I am so happy today but also a little scared about tomorrow
```

**Output:**

```
=====EMOTIONS PERCENTAGE=====

percentage of happy: 50.0%
percentage of fearful: 50.0%

=====SENTIMENTS PERCENTAGE=====

percentage of positive: 100.0%
```

Two bar graphs will also pop up — one for emotions, one for sentiment.

---

## Example Inputs to Try

```
I feel devastated and hopeless after what happened.
```
```
This is amazing! I am so excited and proud of myself.
```
```
I don't know what to do, I feel lost and confused.
```

---

## How It Works

### Emotion Detection
- Input text is cleaned (lowercased, punctuation removed)
- Stop words are filtered out
- Remaining words are matched against `emotions.txt` — a dictionary of 300+ words mapped to emotions
- A `Counter` tracks how many words matched each emotion

### Sentiment Analysis
- Cleaned words are joined back into a string
- `CountVectorizer` converts text to a feature vector
- `MultinomialNB` (Naive Bayes) predicts: **positive**, **negative**, or **neutral**
- Model is trained on `combined_sentiment_data.csv` and cached with `joblib`

### Visualization
- `matplotlib` generates bar charts for emotion distribution and sentiment distribution

---

## Limitations

- Emotion detection is dictionary-based — works best on direct/clear language
- Model accuracy depends on the quality and balance of training CSV data
- Multi-word phrases (e.g., "in a huff") may not match correctly after tokenization
- Does not support non-English text

---

## Real-World Applications

- Mental health journaling tools
- Customer feedback analysis
- Social media sentiment monitoring
- Educational NLP demos

---

## Author

**Name:** Krishna  
**Registration No:** 25BCE10430  
**Branch:** B.Tech CSE (Core)  
**University:** VIT Bhopal University  
**Course:** CSA2001 - Fundamentals of AI and ML

---

## License

This project is submitted as part of the BYOP (Bring Your Own Project) assignment for academic purposes.
