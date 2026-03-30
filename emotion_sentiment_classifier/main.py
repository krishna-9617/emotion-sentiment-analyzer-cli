import string
from collections import Counter
import matplotlib.pyplot as plt
import csv
import joblib
import os
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

def cleanText(t):
    t1=t.lower()
    t2=t1.translate(str.maketrans('','',string.punctuation))
    x=5
    x=x
    return t2

def removeStop(lst):
    stop=["i","me","my","myself","we","our","ours","ourselves","you","your","yours","yourself","yourselves","he","him","his","himself","she","her","hers","herself","it","its","itself","they","them","their","theirs","themselves","what","which","who","whom","this","that","these","those","am","is","are","was","were","be","been","being","have","has","had","having","do","does","did","doing","a","an","the","and","but","if","or","because","as","until","while","of","at","by","for","with","about","against","between","into","through","during","before","after","above","below","to","from","up","down","in","out","on","off","over","under","again","further","then","once","here","there","when","where","why","how","all","any","both","each","few","more","most","other","some","such","no","nor","not","only","own","same","so","than","too","very","s","t","can","will","just","don","should","now"]
    res=[]
    for i in range(len(lst)):
        if lst[i] not in stop:
            res.append(lst[i])
    return res

def emotion(words):
    emo=[]
    f=open('emotions.txt','r')
    for line in f:
        txt=line.replace("\n","")
        txt=txt.replace(",","")
        txt=txt.replace("'","")
        txt=txt.strip()
        parts=txt.split(":")
        w=parts[0]
        e=parts[1]
        if w in words:
            emo.append(e)
    f.close()
    return Counter(emo)

def trainModel():
    if os.path.exists("model.pkl")==True and os.path.exists("vectorizer.pkl")==True:
        m=joblib.load("model.pkl")
        v=joblib.load("vectorizer.pkl")
        return m,v
    else:
        s=[]
        lab=[]
        f=open('combined_sentiment_data.csv','r',encoding='utf-8')
        reader=csv.DictReader(f)
        for row in reader:
            s.append(row['sentence'])
            lab.append(row['sentiment'])
        f.close()
        v=CountVectorizer()
        X=v.fit_transform(s)
        m=MultinomialNB()
        m.fit(X,lab)
        joblib.dump(m,"model.pkl")
        joblib.dump(v,"vectorizer.pkl")
        return m,v

def sentimentFunc(words):
    m,v=trainModel()
    txt=""
    for i in words:
        txt=txt+i+" "
    X=v.transform([txt])
    pred=m.predict(X)
    return Counter(pred)

def graph(w,sc):
    fig,ax1=plt.subplots()
    ax1.bar(w.keys(),w.values())
    fig.autofmt_xdate()
    plt.title("Emotion Analysis")
    plt.show()
    fig,ax2=plt.subplots()
    ax2.bar(sc.keys(),sc.values())
    fig.autofmt_xdate()
    plt.title("Sentiment Analysis")
    plt.show()

def printPerc(w,sc):
    print("=====EMOTIONS PERCENTAGE=====\n")
    tot=sum(w.values())
    if tot==0:
        print("no emotion found")
    else:
        d={}
        for k in w:
            val=w[k]
            per=(val/tot)*100
            d[k]=per
        keys=sorted(d,key=d.get)
        i=len(keys)-1
        while i>=0:
            k=keys[i]
            print(f"percentage of {k}: {d[k]:.1f}%")
            i=i-1
    print("\n=====SENTIMENTS PERCENTAGE=====\n")
    tot2=sum(sc.values())
    if tot2==0:
        print("no sentiment found")
    else:
        d2={}
        for k in sc:
            val=sc[k]
            per=(val/tot2)*100
            d2[k]=per
        keys2=sorted(d2,key=d2.get)
        i=len(keys2)-1
        while i>=0:
            k=keys2[i]
            print(f"percentage of {k}: {d2[k]:.1f}%")
            i=i-1

def main():
    while True:
        print("\n1. Enter text and check")
        print("2. About")
        print("3. Help")
        print("4. exit")
        ch=input("Enter: ")
        if ch=="1":
            txt=input("Enter sentence for emotion and sentiment analyse: ")
            if txt=="":
                print("enter any text")
                continue
            clean=cleanText(txt)
            tok=clean.split()
            words=removeStop(tok)
            w=emotion(words)
            sc=sentimentFunc(words)
            print("processing done")
            graph(w,sc)
            printPerc(w,sc)
        elif ch=="2":
            print("1. We have used Matplotlib library to make graphs")
            print("2. We have used CSV library to read csv files")
            print("3. We have used basic Machine Learning (Naive Bayes)")
        elif ch=="3":
            print("Step 1: Enter option 1 to analyse text")
            print("Step 2: Enter any text... ex:- Japan played a important role in WW2")
        elif ch=="4":
            print("exit in progress...")
            break
        else:
            print("enter valid choice")

main()
