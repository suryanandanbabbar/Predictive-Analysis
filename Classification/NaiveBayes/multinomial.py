from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Sample Dataset
emails = [
    "Win money now free offer",
    "Congratulations claim your prize",
    "Important meeting at office",
    "Schedule for project discussion",
    "Free entry to win rewards",
    "Urgent response needed free"
]

labels = [1, 1, 0, 0, 1, 1] # 1 = spam, 0 = not spam

# Convert text to numeric features
"""
What does CountVectorizer do?
CountVectorizer converts a collection of text documents to a matrix of token counts.
It tokenizes the text, counts the occurrences of each word, and creates a sparse matrix representation
of the document-term matrix.

It does "TEXT-MINING" by transforming text data into a format suitable for machine learning algorithms.
- Everything in small letters, no punctuation, and each word is treated as a feature.
- Occurrences of words are counted, and the result is a matrix where each row represents a document
    and each column represents a word.

"""
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=0)

# Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print(model.predict(vectorizer.transform(["Free offer"])))