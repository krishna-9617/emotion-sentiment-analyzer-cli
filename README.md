# Emotion & Sentiment Analyzer
So basically this is a CLI tool I made that takes any text you type and tries to figure out what emotions and sentiment are in it. It uses Python, some NLP stuff, and a basic ML model under the hood.

---

## What It Does

You just type a sentence and the tool tells you:
- What **emotions** are there — like happy, angry, sad, fearful, etc.
- What **sentiment** it has — positive, negative, or neutral
- And it also shows **bar graphs** for both, so you can actually see it visually

No internet needed at all. The whole thing runs in the terminal.

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

> **Note:** First time you run it, the model trains itself from `combined_sentiment_data.csv` and saves as `model.pkl`. After that it just loads from cache, so it's pretty fast.

---

## How to Use

When you run it, a simple menu comes up:

```
1. Enter text and check
2. About
3. Help
4. exit
```

**Step 1** → Pick option `1`  
**Step 2** → Type whatever sentence you want, like:

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

Two bar graphs will pop up too — one showing emotions, one showing sentiment.

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
- First the input gets cleaned — lowercased, punctuation removed
- Stop words are filtered out (basically words like "the", "is", "a" that don't really mean anything useful)
- The remaining words are matched against `emotions.txt` which is kind of a dictionary with 300+ words, each mapped to an emotion
- A `Counter` keeps track of how many words matched each emotion

### Sentiment Analysis
- The cleaned words get joined back into a string
- `CountVectorizer` turns that into a feature vector
- Then `MultinomialNB` (Naive Bayes) predicts if it's **positive**, **negative**, or **neutral**
- The model is trained on `combined_sentiment_data.csv` and saved using `joblib` so it doesn't retrain every single time

### Visualization
- `matplotlib` just draws bar charts — one for emotions, one for sentiment

---

## Limitations

Honestly there are a few things this tool doesn't handle that well:

- Since emotion detection is dictionary-based, it works fine for simple clear sentences but might miss sarcasm or indirect language
- The sentiment model's accuracy kind of depends on how good and balanced the training data is
- Multi-word phrases like "in a huff" probably won't get detected properly after tokenization breaks them up
- Non-English text won't work at all

---

## Real-World Applications

Even with the limitations, I think this kind of tool could actually be useful for things like:

- Mental health journaling apps
- Analyzing customer feedback
- Social media sentiment tracking
- Just learning how NLP works in general

---

## Author

**Name:** Krishna  
**Registration No:** 25BCE10430  
**Branch:** B.Tech CSE (Core)  
**University:** VIT Bhopal University  
**Course:** CSA2001 - Fundamentals of AI and ML

---

## License

This project was made for the BYOP (Bring Your Own Project) assignment as part of my academics.
