# Text Clustering Mini Project

## Author
Pragna Madineni Jayadev

## Overview
This project performs clustering on multiple real-world text datasets using different unsupervised machine learning algorithms. The objective is to analyze how clustering techniques behave on high-dimensional textual data.

## Algorithms Used
- KMeans Clustering  
- Hierarchical Clustering (Agglomerative)  
- DBSCAN  

## Datasets Used
- 20 Newsgroups (news articles)  
- IMDb Movie Reviews  
- Sentiment140 Tweets  

## Methodology
The project follows these steps:
1. Text preprocessing (cleaning, stopword removal)  
2. Feature extraction using TF-IDF  
3. Application of clustering algorithms  
4. Evaluation using silhouette score  
5. Visualization using PCA and t-SNE  

## Results
- KMeans performed relatively better across all datasets  
- Hierarchical clustering showed inconsistent results  
- DBSCAN failed due to sparsity in TF-IDF data  
- Overall silhouette scores were low, indicating overlapping clusters  

## Outputs
- PCA and t-SNE visualizations  
- Dendrogram (Hierarchical clustering)  
- Elbow method graph  
- Final comparison table (clustering_results.csv)  

## How to Run
1. Install dependencies:
pip install pandas numpy scikit-learn matplotlib  

2. Update dataset paths in clustering.py  

3. Run:
python clustering.py  

## Key Insight
Clustering textual data is challenging due to sparsity and overlapping feature spaces. Among the evaluated methods, KMeans provided the most consistent performance.

## Files
- clustering.py  
- clustering_results.csv  
- README.md  

## Conclusion
This project demonstrates the limitations of traditional clustering algorithms on text data and highlights the importance of feature representation in text mining tasks.
