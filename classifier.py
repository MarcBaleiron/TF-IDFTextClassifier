from collections import Counter
import math

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


# Tests
doc1 = ["great", "film", "great", "acting"]
doc2 = ["terrible", "film", "boring", "acting"]
doc3 = ["great", "story"]

documents = [doc1, doc2, doc3]
idf = compute_idf(documents)
tfidf_doc1 = compute_tfidf(doc1, idf)
print(tfidf_doc1)
tfidf_doc2 = compute_tfidf(doc2, idf)
print(tfidf_doc2)
tfidf_doc3 = compute_tfidf(doc3, idf)
print(tfidf_doc3)