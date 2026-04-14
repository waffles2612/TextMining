#ablation study using SVM
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC

# Load data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

X_train_full = train['text']
y_train = train['label']

X_test_full = test['text']
y_test = test['label']

# --------- Create SHORT TEXT version ---------
X_train_short = X_train_full.str.split().str[:10].str.join(" ")
X_test_short = X_test_full.str.split().str[:10].str.join(" ")

# --------- Function to train and evaluate ---------
def run_experiment(vectorizer, X_train, X_test, name):
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LinearSVC()
    model.fit(X_train_vec, y_train)

    preds = model.predict(X_test_vec)
    acc = accuracy_score(y_test, preds)

    print(f"{name} Accuracy: {acc:.4f}")
    return acc

# --------- Experiments ---------

results = []

# 1. TF-IDF (Full Text)
acc = run_experiment(
    TfidfVectorizer(stop_words='english', max_features=5000),
    X_train_full, X_test_full,
    "TF-IDF + Full Text"
)
results.append(("TF-IDF", "Full", acc))

# 2. CountVectorizer (Full Text)
acc = run_experiment(
    CountVectorizer(stop_words='english', max_features=5000),
    X_train_full, X_test_full,
    "Count + Full Text"
)
results.append(("Count", "Full", acc))

# 3. TF-IDF (Short Text)
acc = run_experiment(
    TfidfVectorizer(stop_words='english', max_features=5000),
    X_train_short, X_test_short,
    "TF-IDF + Short Text"
)
results.append(("TF-IDF", "Short", acc))

# 4. CountVectorizer (Short Text)
acc = run_experiment(
    CountVectorizer(stop_words='english', max_features=5000),
    X_train_short, X_test_short,
    "Count + Short Text"
)
results.append(("Count", "Short", acc))

# --------- Print Summary Table ---------
print("\nFinal Results:")
for vec, text_type, acc in results:
    print(f"{vec} + {text_type}: {acc:.4f}")

import matplotlib.pyplot as plt

labels = [
    "TF-IDF Full",
    "Count Full",
    "TF-IDF Short",
    "Count Short"
]

accuracies = [acc for _, _, acc in results]

plt.figure(figsize=(8,5))  # better size

plt.bar(labels, accuracies)
plt.title("Ablation Study Comparison")
plt.xlabel("Configuration")
plt.ylabel("Accuracy")
plt.xticks(rotation=30)

plt.grid()
plt.tight_layout()

plt.savefig("ablation_graph.png")
plt.show()
