import string
from collections import Counter
import matplotlib.pyplot as plt
import csv
import joblib
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

def clean_text(text):
    lower_case = text.lower()
    cleaned_text = lower_case.translate(str.maketrans('', '', string.punctuation))
    return cleaned_text

def remove_stopwords(tokenised_text):
    stop_words = ["i","me","my","myself","we","our","ours","ourselves","you","your","yours","yourself",
                  "yourselves","he","him","his","himself","she","her","hers","herself","it","its","itself",
                  "they","them","their","theirs","themselves","what","which","who","whom","this","that","these",
                  "those","am","is","are","was","were","be","been","being","have","has","had","having","do",
                  "does","did","doing","a","an","the","and","but","if","or","because","as","until","while",
                  "of","at","by","for","with","about","against","between","into","through","during","before",
                  "after","above","below","to","from","up","down","in","out","on","off","over","under","again",
                  "further","then","once","here","there","when","where","why","how","all","any","both","each",
                  "few","more","most","other","some","such","no","nor","not","only","own","same","so","than",
                  "too","very","s","t","can","will","just","don","should","now"]
    final_words = []
    for word in tokenised_text:
        if word not in stop_words:
            final_words.append(word)
    return final_words

def emotion(final_words):
    emotions_list = []
    with open('emotions.txt', 'r') as file:
        for line in file:
            clear_text = line.replace("\n", "").replace(",", "").replace("'", "").strip()
            word, emotion = clear_text.split(":")
            if word in final_words:
                emotions_list.append(emotion)
    return Counter(emotions_list)

def train_model():
    if os.path.exists("model.pkl") and os.path.exists("vectorizer.pkl"):
        model = joblib.load("model.pkl")
        vectorizer = joblib.load("vectorizer.pkl")
        return model, vectorizer
    else:
        sentences = []
        sentiments = []

        with open('combined_sentiment_data.csv', 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                sentences.append(row['sentence'])
                sentiments.append(row['sentiment'])

        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(sentences)

        model = MultinomialNB()
        model.fit(X, sentiments)

        joblib.dump(model, "model.pkl")
        joblib.dump(vectorizer, "vectorizer.pkl")

        return model, vectorizer

def sentiment_func(final_words):
    model, vectorizer = train_model()

    text = " ".join(final_words)
    X = vectorizer.transform([text])

    prediction = model.predict(X)

    return Counter(prediction)

def graph(w, sentiment_counts): 

    fig, ax1 = plt.subplots()
    ax1.bar(w.keys(), w.values())
    fig.autofmt_xdate()
    plt.title("Emotion Analysis")
    plt.show() 

    fig, ax2 = plt.subplots()
    ax2.bar(sentiment_counts.keys(), sentiment_counts.values())
    fig.autofmt_xdate()
    plt.title("Sentiment Analysis")
    plt.show()

def print_percentages(w, sentiment_counts):
    print("=====EMOTIONS PERCENTAGE=====\n")
    total = sum(w.values())
    if total == 0:
        print("no emotion found")
    else:
        percent_dict = {}
        for key, value in w.items():
            percent = (value/total)*100
            percent_dict[key] = percent
        sorted_keys = sorted(percent_dict, key=percent_dict.get)

        for i in range(len(sorted_keys)-1, -1, -1):
            k = sorted_keys[i]
            print(f"percentage of {k}: {percent_dict[k]:.1f}%")

    print("\n=====SENTIMENTS PERCENTAGE=====\n")
    total2 = sum(sentiment_counts.values())
    if total2 == 0:
        print("no sentiment found")
    else:
        percent_dict2 = {}
        for key, value in sentiment_counts.items():
            percent = (value/total2)*100
            percent_dict2[key] = percent
        sorted_keys2 = sorted(percent_dict2, key=percent_dict2.get)

        for i in range(len(sorted_keys2)-1, -1, -1):
            k = sorted_keys2[i]
            print(f"percentage of {k}: {percent_dict2[k]:.1f}%")

def main():
    while True:
        print("\n1. Enter text and check")
        print("2. About")
        print("3. Help")
        print("4. exit")
        choice = input("Enter: ")

        if choice == "1":
            text = input("Enter sentence for emotion and sentiment analyse: ")
            if text == "":
                print("enter any text")
                continue

            cleaned_text = clean_text(text)
            tokenised_text = cleaned_text.split()
            final_words = remove_stopwords(tokenised_text)
            w = emotion(final_words)
            sentiment_counts = sentiment_func(final_words)

            graph(w, sentiment_counts)
            print_percentages(w, sentiment_counts)

        elif choice == "2":
            print("1. We have used Matplotlib library to make graphs")
            print("2. We have used CSV library to read csv files")
            print("3. We have used basic Machine Learning (Naive Bayes)")

        elif choice == "3":
            print("Step 1: Enter option 1 to analyse text")
            print("Step 2: Enter any text... ex:- Japan played a important role in WW2")

        elif choice == "4":
            print("exit in progress...")
            break
        
        else:
            print("enter valid choice")

main()
