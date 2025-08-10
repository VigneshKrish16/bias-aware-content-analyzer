import numpy as np
from sklearn.metrics import mutual_info_score
from collections import Counter

class BiasDetector:
    def __init__(self):
        self.inclusivity_weight = 0.4
        self.diversity_weight = 0.3
        self.fairness_weight = 0.3
    
    def calculate_inclusivity(self, groups):
        group_counts = Counter(groups)
        total = sum(group_counts.values())
        if total == 0:
            return 0
        
        max_count = max(group_counts.values())
        inclusivity = 0
        for count in group_counts.values():
            inclusivity += (count / total) * (1 - (count / max_count))
        return inclusivity
    
    def calculate_diversity(self, clusters, groups):
        if len(clusters) < 2 or len(groups) < 2:
            return 0
        
        # Normalized Mutual Information
        h_cluster = self._entropy(clusters)
        h_group = self._entropy(groups)
        mi = mutual_info_score(clusters, groups)
        
        if h_cluster + h_group == 0:
            return 0
        return 2 * mi / (h_cluster + h_group)
    
    def calculate_fairness(self, outcomes, protected_attrs):
        if len(outcomes) == 0:
            return 1
        
        # Disparate Impact ratio
        privileged_outcome = sum(1 for o, a in zip(outcomes, protected_attrs) if a == 1 and o == 1)
        privileged_total = sum(1 for a in protected_attrs if a == 1)
        
        unprivileged_outcome = sum(1 for o, a in zip(outcomes, protected_attrs) if a == 0 and o == 1)
        unprivileged_total = sum(1 for a in protected_attrs if a == 0)
        
        if privileged_total == 0 or unprivileged_total == 0:
            return 1
            
        ratio1 = (privileged_outcome / privileged_total) / (unprivileged_outcome / unprivileged_total)
        ratio2 = 1 / ratio1
        return min(ratio1, ratio2)
    
    def _entropy(self, labels):
        counts = np.bincount(labels)
        probs = counts / len(labels)
        return -np.sum([p * np.log2(p) for p in probs if p > 0])
    
    def calculate_metrics(self, documents):
        # Extract features for metrics calculation
        groups = [doc.get('demographic_group', 0) for doc in documents]
        clusters = [doc.get('cluster_id', 0) for doc in documents]
        outcomes = [doc.get('outcome', 1) for doc in documents]
        protected_attrs = [doc.get('protected_attribute', 0) for doc in documents]
        
        inclusivity = self.calculate_inclusivity(groups)
        diversity = self.calculate_diversity(clusters, groups)
        fairness = self.calculate_fairness(outcomes, protected_attrs)
        
        biq = (self.inclusivity_weight * inclusivity + 
               self.diversity_weight * diversity + 
               self.fairness_weight * fairness)
        
        return {
            'biq': biq,
            'cfs': fairness,  # Content Fairness Score
            'cds': diversity,  # Content Diversity Score
            'inclusivity': inclusivity,
            'diversity': diversity,
            'fairness': fairness
        }