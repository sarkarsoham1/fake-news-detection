# Fake News Detection

A machine learning-based NLP project that classifies news articles as **Real** or **Fake** using the WELFake dataset. The project covers text preprocessing, TF-IDF feature extraction, multiple classification models, and model evaluation.

## Project Overview

The objective is to automatically classify news articles into two categories:

- `0` — Real News
- `1` — Fake News

The project combines the **title** and **article text** into a single content feature and uses traditional machine learning techniques for text classification.

## Dataset

The project uses the **WELFake dataset**, containing approximately **72,000 news articles** with real/fake labels.

The dataset is relatively balanced:

- Real News: ~48.6%
- Fake News: ~51.4%

## Workflow

```text
Dataset
   ↓
Data Cleaning & EDA
   ↓
Combine Title + Text
   ↓
Text Preprocessing
   ↓
Train/Test Split
   ↓
TF-IDF Feature Extraction
   ↓
Model Training
   ├── Logistic Regression
   ├── Multinomial Naive Bayes
   └── Random Forest
   ↓
Model Evaluation & Comparison
```

## Text Preprocessing

The text preprocessing pipeline includes:

1. Handling missing values
2. Combining title and article text
3. Removing non-alphabetic characters using regular expressions
4. Converting text to lowercase
5. Tokenization using `split()`
6. Stopword removal
7. Lemmatization using `WordNetLemmatizer`

## Exploratory Data Analysis

EDA was performed to understand:

- Class distribution
- Missing values
- Text/title length patterns
- Important textual patterns

## TF-IDF Feature Extraction

The processed text is converted into numerical features using **TF-IDF (Term Frequency-Inverse Document Frequency)**.

Configuration:

```python
TfidfVectorizer(
    max_features=5000,
    min_df=2,
    max_df=0.8,
    ngram_range=(1, 2)
)
```

This uses up to **5,000 features**, including both unigrams and bigrams.

To avoid data leakage, the vectorizer is fitted only on the training data and then used to transform the test data:

```python
X_train = tfidf.fit_transform(X_train_text)
X_test = tfidf.transform(X_test_text)
```

## Models

### Logistic Regression

Used as a strong baseline for high-dimensional TF-IDF text features.

**Test Accuracy: 94.68%**

### Multinomial Naive Bayes

Used as a traditional and computationally efficient text-classification baseline.

It performed lower than Logistic Regression on the test set.

### Random Forest

Used to evaluate whether an ensemble tree-based model could improve classification performance.

**Test Accuracy: 95.42%**

Random Forest achieved the best test accuracy among the evaluated models.

## Model Performance

| Model | Test Accuracy |
|---|---:|
| Logistic Regression | 94.68% |
| Multinomial Naive Bayes | Lower than Logistic Regression |
| Random Forest | **95.42%** |

### Random Forest Classification Performance

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Real | 0.97 | 0.93 | 0.95 |
| Fake | 0.94 | **0.98** | **0.96** |

The Random Forest model achieved **98% recall for fake news**, meaning it correctly identified approximately 98% of the fake articles in the test set.

### Confusion Matrix

```text
                 Predicted
              Real      Fake
Actual Real   9796       713
       Fake    278     10854
```

## Key Takeaways

- TF-IDF provided an effective numerical representation of the news articles.
- Logistic Regression achieved strong performance with a 94.68% test accuracy.
- Multinomial Naive Bayes performed below Logistic Regression.
- Random Forest achieved the best test accuracy at **95.42%**.
- Random Forest achieved **98% recall for fake news** and a **0.96 F1-score** for the fake class.
- The higher Random Forest training accuracy (~99.33%) compared with test accuracy indicates some overfitting.

## Technologies Used

- Python
- Pandas
- NumPy
- NLTK
- Scikit-learn
- Matplotlib
- Seaborn
- TF-IDF
- Logistic Regression
- Multinomial Naive Bayes
- Random Forest

## Future Improvements

- Hyperparameter tuning and cross-validation
- Testing Linear SVM and other classifiers
- Experimenting with Word2Vec or other embeddings
- Fine-tuning transformer models such as BERT
- Evaluating performance on completely unseen news sources
- Monitoring concept drift as news topics and writing patterns change

## Conclusion

This project demonstrates an end-to-end NLP classification workflow, from data cleaning and exploratory analysis to text preprocessing, TF-IDF feature engineering, model training, and evaluation. Among the evaluated models, **Random Forest achieved the strongest test performance with 95.42% accuracy and 98% recall for fake news**.
