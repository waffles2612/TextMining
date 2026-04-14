
---

## 📌 Topic Modelling

### Overview
Comparative study of three unsupervised topic modeling techniques applied to a corpus of text documents to discover hidden thematic structures without labeled data.

### Algorithms Implemented
| Algorithm | Type | Library |
|---|---|---|
| Latent Dirichlet Allocation (LDA) | Probabilistic Generative Model | Gensim |
| Non-negative Matrix Factorization (NMF) | Matrix Factorization | Scikit-learn |
| Latent Semantic Analysis (LSA) | Singular Value Decomposition | Scikit-learn |

### Evaluation Metrics
- **LDA** → C_v Coherence Score (higher is better)
- **NMF** → Reconstruction Error (lower is better)
- **LSA** → Explained Variance Ratio (higher is better)

### Ablation Study
The ablation study systematically evaluates the effect of:
- Number of topics (2 → 6)
- Preprocessing configurations (stopword removal, punctuation filtering)
- Vocabulary size (200 → 2000 features)
- Fixed model comparison across all three approaches

---

## 🛠️ Installation

```bash
pip install gensim nltk scikit-learn matplotlib
```

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
```

---

## 🚀 Usage

1. Place your `.txt` documents inside the `sample_data/` folder
2. Run `topic_modelling.py` in Google Colab or Jupyter Notebook
3. Output will display discovered topics for each model along with comparison charts

---

## 📊 Results Summary

| Model | Speed | Interpretability | Accuracy |
|---|---|---|---|
| LDA | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| NMF | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| LSA | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |

> NMF produces the most interpretable topics, LSA is the fastest, and LDA offers the most robust probabilistic framework.

---

## 🧰 Technologies Used

- Python 3.x
- Gensim
- Scikit-learn
- NLTK
- Matplotlib
- Google Colab

---

## 👩‍💻 Author

**waffles2612**  
[GitHub Profile](https://github.com/waffles2612)

---

## 📜 License

This project is for academic purposes only.
