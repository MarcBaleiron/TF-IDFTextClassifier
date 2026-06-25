# TF-IDFTextClassifier
A Python sentiment analysis tool that uses TF-IDF vectorization and Logistic Regression to classify text reviews as positive or negative.


## Features
- **TF-IDF Vectorization**: Converts raw review text into weighted numerical features based on term frequency and inverse document frequency
- **Two Vectorization Modes**: Choose between a from-scratch TF-IDF implementation or scikit-learn's optimized `TfidfVectorizer`
- **Stopword Filtering**: Removes common, low-signal English words before vectorizing (manual mode only)
- **Logistic Regression Classifier**: Trains a binary sentiment classifier (positive/negative) on the vectorized reviews
- **Train/Test Split Evaluation**: Automatically splits the dataset and reports accuracy on held-out data

## Installation
Requires Python 3.6+ with the following dependencies:
```bash
pip install pandas scikit-learn
```


## Quick Start
```bash
# Evaluate sentiment classification using the manual TF-IDF implementation
python classifier.py reviews.csv --function evaluate

# Evaluate sentiment classification using scikit-learn's TfidfVectorizer
python classifier.py reviews.csv --function evaluate-sklearn
```


## Usage
### Command-Line Interface
```bash
python classifier.py <filepath> [--function <function_name>]
```

### Options
- `filepath`: Path to the CSV file containing reviews to analyze (required)
- `--function, -f`: Analysis function to run (default: `evaluate`)

### Input Format
A CSV file with at least two columns:
- `review`: The raw text of the review
- `sentiment`: The label associated with each review (e.g. positive/negative)

### Available Functions

**`evaluate`** (default) — Classify sentiment using a TF-IDF implementation built from scratch
```bash
python classifier.py reviews.csv
```

**`evaluate-sklearn`** — Classify sentiment using scikit-learn's `TfidfVectorizer`
```bash
python classifier.py reviews.csv --function evaluate-sklearn
```


## How It Works
### Preprocessing
1. **Loading**: Reads the CSV file into a DataFrame and extracts the `review` and `sentiment` columns
2. **Cleaning**: Lowercases text and strips `<br /><br />` line-break artifacts
3. **Tokenization**: Extracts words, contractions, abbreviations, and decimal numbers
4. **Stopword Removal**: Filters out common English stopwords and non-word tokens (manual mode only)
### Vectorization
- **Manual mode**: Computes term frequency (TF) per document and inverse document frequency (IDF) across the training set, then multiplies them to produce TF-IDF scores for each token in a fixed vocabulary
- **Sklearn mode**: Fits scikit-learn's `TfidfVectorizer` on the training data and transforms both train and test sets
### Training and Evaluation
1. **Splitting**: Divides the dataset into 80% training and 20% testing data (`random_state=42`)
2. **Training**: Fits a `LogisticRegression` model on the vectorized training reviews and their sentiment labels
3. **Prediction**: Predicts sentiment labels for the held-out test reviews
4. **Scoring**: Prints the classifier's accuracy on the test set


## Function Reference
### Core Functions
- `tokenize_words(text)` — Tokenizes text into words, contractions, abbreviations, and decimals
- `clean(text)` — Lowercases text and removes `<br /><br />` artifacts
- `compute_tf(tokens)` — Returns a dictionary of term frequencies for a single document
- `compute_idf(documents)` — Returns a dictionary of inverse document frequencies across a corpus
- `compute_tfidf(tokens, idf)` — Combines TF and IDF into a TF-IDF score dictionary for a document
### Analysis Functions
- `evaluate(filepath)` — Trains and evaluates a Logistic Regression classifier using a manually computed TF-IDF matrix, then prints accuracy
- `evaluate_sklearn(filepath)` — Trains and evaluates a Logistic Regression classifier using scikit-learn's `TfidfVectorizer`, then prints accuracy

## Example Output

### `--function evaluate`
```
Accuracy: 0.8672
```

### `--function evaluate-sklearn`
```
Accuracy: 0.8915
```


## Notes
- All text processing is case-insensitive
- Stopwords (e.g. "the", "and", "is") are excluded only in the manual TF-IDF pipeline; scikit-learn's `TfidfVectorizer` uses its own internal handling
- The manual implementation builds its vocabulary strictly from the training set, so tokens unseen during training contribute a TF-IDF score of 0 at test time
- The `evaluate-sklearn` mode is typically faster and may yield different accuracy than the manual implementation, since `TfidfVectorizer` uses smoothed IDF and L2 normalization by default
- The train/test split uses a fixed `random_state=42` for reproducibility
- Short form: `-f` for `--function`