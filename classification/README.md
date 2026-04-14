**Text Classification (News Dataset)**

This module implements and evaluates multiple machine learning models for news text classification using TF-IDF features. It also includes an ablation study to analyze how vectorization methods and text length impact performance.

The objective of this project is to classify news articles into predefined categories using traditional Natural Language Processing (NLP) techniques. The project compares different models and studies the effect of feature extraction choices.

**Methodology**

1. Data
* Input files:

  * train_sample.csv
  * test_sample.csv
* Each dataset contains:

  * text (news article)
  * label (category)
Note: A smaller sample of the original dataset is included in this repository for demonstration purposes due to file size limitations.

2. Feature Extraction
* TF-IDF Vectorization:
  * Removes English stopwords
  * Limits vocabulary to top 5000 features

* Count Vectorization:
  * Used for comparison in ablation study

3. Models Implemented
The following models are trained using TF-IDF features:
* Multinomial Naive Bayes
* Logistic Regression
* Support Vector Machine (Linear SVM)

**Model Evaluation**

Each model is evaluated using:
* Accuracy Score
* Precision, Recall, F1-score

A comparison graph is generated:
model_accuracy.png: Displays accuracy of all models

**Ablation Study**

The ablation study evaluates how different design choices affect performance.

Variables Tested:
* Vectorization Method:
  * TF-IDF
  * Count Vectorizer

* Text Length
  * Full text
  * Short text (first 10 words only)

**Output**

* Accuracy printed for each experiment
* Graph saved as: ablation_graph.png

**Project Structure**

```bash
classification/
│
├── classifiers_news.py
├── ablation_study.py
├── conversion.py (Converts original dataset from parquet format to CSV)
├── train_sample.csv
├── test_sample.csv
├── model_accuracy.png
├── ablation_graph.png
├── requirements.txt
└── README.md
```
**How to Run**

1. Install Dependencies
```bash
pip install pandas scikit-learn matplotlib
```
2. Run Classification Models
```bash
python classifiers_news.py
```
3. Run Ablation Study
```bash
python ablation_study.py
```

**Key Insights**

* TF-IDF generally performs better than Count Vectorizer due to better weighting of important terms.
* Support Vector Machines (SVM) typically achieve higher accuracy in text classification tasks.
* Reducing text length negatively impacts performance, highlighting the importance of context.

**Conclusion**

This module demonstrates how classical machine learning models combined with TF-IDF features can effectively perform text classification. It also highlights how feature representation and input size influence model performance.
