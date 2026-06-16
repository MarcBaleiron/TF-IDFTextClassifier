import re
from collections import Counter
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import math


# List of words that do not need to be count (from nltk)
STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours',
    'you', 'your', 'yours', 'he', 'him', 'his', 'she',
    'her', 'it', 'its', 'they', 'them', 'what', 'which',
    'who', 'this', 'that', 'am', 'is', 'are', 'was',
    'were', 'be', 'have', 'has', 'had', 'do', 'does',
    'did', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
    'because', 'as', 'until', 'while', 'of', 'at', 'by',
    'for', 'with', 'about', 'against', 'between', 'into',
    'through', 'during', 'before', 'after', 'above',
    'below', 'to', 'from', 'up', 'down', 'in', 'out',
    'on', 'off', 'over', 'under', 'again', 'further',
    'then', 'once', 'here', 'there', 'when', 'where',
    'why', 'how', 'all', 'any', 'both', 'each', 'few',
    'more', 'most', 'other', 'some', 'such', 'no', 'nor',
    'not', 'only', 'own', 'same', 'so', 'than', 'too',
    'very', 'can', 'will', 'just', 'should', 'now'
}


# Helper functions
def tokenize_words(text):
    return re.findall(r"(?:[A-Z]\.)+|\d+\.\d+|\w+(?:'\w+)?|[^\w\s]", text)

def clean(text):
    text = text.lower()
    text = text.replace("<br /><br />", "")
    return text

def compute_tf(tokens):
    tf_dictionary = {}
    unique_tokens = Counter(tokens)
    for t in unique_tokens:
        count_of_word = unique_tokens[t]
        total_words = len(tokens)
        tf_dictionary[t] = count_of_word / total_words

    return tf_dictionary

def compute_idf(documents):
    idf_dictionary = {}
    total_docs = len(documents)
    doc_count = {}
    for d in documents:
        unique_words_in_d = set(d)
        for w in unique_words_in_d:
            doc_count[w] = doc_count.get(w, 0) + 1

    for w in doc_count:
            idf_dictionary[w] = math.log(total_docs / doc_count[w])

    return idf_dictionary

def compute_tfidf(tokens, idf):
    tfidf_dictionary = {}
    tf = compute_tf(tokens)
    for t in tf:
        tfidf_dictionary[t] = tf[t] * idf.get(t, 0)

    return tfidf_dictionary


# Function that evaluates the reviews and returns its accuracy 
#   without using sklearn's "TfidfVectorizer"
def evaluate(filepath):
    df = pd.read_csv(filepath)
    documents = df["review"].apply(clean)
    labels = df["sentiment"]

    # Split the data
    x = documents
    y = labels
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    # Vectorize the data
    tokenized_train = x_train.apply(tokenize_words)
    filtered_train = tokenized_train.apply(lambda tokens: [w for w in tokens if re.search(r"\w", w) and w not in STOPWORDS])
    idf = compute_idf(filtered_train)

    vocabulary = list(idf.keys())

    x_train_vectorized = []
    for tokens in filtered_train:
        tfidf = compute_tfidf(tokens, idf)
        x_train_vectorized.append([tfidf.get(word, 0) for word in vocabulary])

    x_test_vectorized = []
    for tokens in x_test.apply(tokenize_words):
        tfidf = compute_tfidf(tokens, idf)
        x_test_vectorized.append([tfidf.get(word, 0) for word in vocabulary])

    # Model training
    model = LogisticRegression()
    model.fit(x_train_vectorized, y_train)

    # Predictions
    y_pred = model.predict(x_test_vectorized)

    # Evaluation
    print("Accuracy:", accuracy_score(y_test, y_pred))

# Function that evaluates the reviews and returns its accuracy
#   using sklearn's "TfidfVectorizer"
def evaluate_sklearn(filepath):
    df = pd.read_csv(filepath)
    documents = df["review"].apply(clean)
    labels = df["sentiment"]

    # Split the data
    x = documents
    y = labels
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    # Vectorize the data    
    vectorizer = TfidfVectorizer()
    x_train_vectorized = vectorizer.fit_transform(x_train)
    x_test_vectorized = vectorizer.transform(x_test)

    # Model training
    model = LogisticRegression()
    model.fit(x_train_vectorized, y_train)

    # Predictions
    y_pred = model.predict(x_test_vectorized)

    # Evaluation
    print("Accuracy:", accuracy_score(y_test, y_pred))


# Command-line arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify sentiment in reviews")
    parser.add_argument("filepath", help="Path to the text file to analyze")
    parser.add_argument("--function", "-f", choices=["evaluate", "evaluate-sklearn"], default="evaluate", help="Analysis function to run")
    
    args = parser.parse_args()
    
    if args.function == "evaluate":
        evaluate(args.filepath)
    elif args.function == "evaluate-sklearn":
        evaluate_sklearn(args.filepath)