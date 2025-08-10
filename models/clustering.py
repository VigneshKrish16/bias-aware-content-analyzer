from sklearn.mixture import GaussianMixture
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
import numpy as np

class ContentClusterer:
    def __init__(self, n_clusters=5, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.gmm = GaussianMixture(n_components=n_clusters, random_state=random_state)
    
    def cluster(self, documents):
        if not documents:
            return []
            
        # Extract text content
        texts = [doc.get('content', '') for doc in documents]
        
        # Vectorize text using TF-IDF
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        
        # Fit GMM model
        self.gmm.fit(tfidf_matrix.toarray())
        
        # Predict clusters
        clusters = self.gmm.predict(tfidf_matrix.toarray())
        probabilities = self.gmm.predict_proba(tfidf_matrix.toarray())
        
        # Calculate diversity score for each document
        diversity_scores = self._calculate_diversity_scores(tfidf_matrix)
        
        # Enhance documents with cluster information
        clustered_docs = []
        for i, doc in enumerate(documents):
            doc['cluster_id'] = int(clusters[i])
            doc['cluster_probability'] = float(np.max(probabilities[i]))
            doc['diversity_score'] = float(diversity_scores[i])
            clustered_docs.append(doc)
            
        return clustered_docs
    
    def _calculate_diversity_scores(self, tfidf_matrix):
        # Calculate pairwise distances between documents
        distances = pairwise_distances(tfidf_matrix, metric='cosine')
        
        # For each document, calculate average distance to other documents
        diversity_scores = np.mean(distances, axis=1)
        
        # Normalize scores to [0, 1]
        if np.max(diversity_scores) > 0:
            diversity_scores = diversity_scores / np.max(diversity_scores)
            
        return diversity_scores