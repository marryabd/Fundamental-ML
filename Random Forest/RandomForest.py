from DecisionTree import DecisionTree
import numpy as np
from collections import Counter


class RandomForest:
    def __init__(self, n_trees=10, n_features=None, max_depth=10, min_samples_split=2):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []
    
    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            X_sample, y_sample = self._bootstrap_samples(X, y)
            tree = DecisionTree(max_depth=self.max_depth, 
                                min_samples_split=self.min_samples_split, 
                                n_features=self.n_features)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
    
    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        # Ensure tree_preds is at least 2D before swapping axes
        # if tree_preds.ndim == 1:
        #     tree_preds = tree_preds[:, np.newaxis]
        swap_tree_preds = np.swapaxes(tree_preds, 0, 1)
        predictions = np.array([self._most_common_label(pred) for pred in swap_tree_preds])
        return predictions
    
    def _bootstrap_samples(self, X, y):
        n_sample = X.shape[0]
        idx = np.random.choice(n_sample, size=n_sample, replace=True)
        return X[idx], y[idx]
        
    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common
        
        
