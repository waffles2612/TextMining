import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# Load data
train = pd.read_csv("train_sample.csv")
test = pd.read_csv("test_sample.csv")

X_train = train['text']
y_train = train['label']

X_test = test['text']
y_test = test['label']

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Models
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=200),
    "SVM": LinearSVC()
}

# Train and evaluate
for name, model in models.items():
    print(f"\n{name}")
    model.fit(X_train_tfidf, y_train)
    preds = model.predict(X_test_tfidf)

    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))

import matplotlib.pyplot as plt

accuracies = {}

# Train + store accuracy (single loop)
for name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    preds = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, preds)
    accuracies[name] = acc

# Plot
plt.figure()
plt.bar(accuracies.keys(), accuracies.values())
plt.title("Model Accuracy Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.grid()

plt.savefig("model_accuracy.png")

plt.show()
