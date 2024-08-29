from collections import Counter
import numpy as np

class Node:
  def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
    self.feature = feature
    self.threshold = threshold
    self.left = left
    self.right = right
    self.value = value

  def is_leaf_node(self):
    return self.value is not None

class DecisionTree:

  def __init__(self, max_depth=100, min_samples_split=2):
    self.max_depth = max_depth
    self.min_samples_split = min_samples_split
    self.root = None
  

  def fit(self, X, y, X_val=None, y_val=None):
    self.n_features = X.shape[1]
    self.root = self._grow_tree(X, y)

    if X_val is not None and y_val is not None:
      self.root = self._prune_tree(self.root, X_val, y_val)
  

  def _grow_tree(self, X, y, depth=0):
    n_samples, n_features = X.shape
    n_labels = len(np.unique(y))

    # stopping criteria 
    if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
      leaf_value = self._most_common_label(y)
      return Node(value=leaf_value)

    # find the best split
    best_feature, best_threshold = self._best_split(X, y, n_features)
    left_idx, right_idx = self._split(X[:, best_feature], best_threshold)

    left = self._grow_tree(X[left_idx, :], y[left_idx], depth+1)
    right = self._grow_tree(X[right_idx, :], y[right_idx], depth+1)

    return Node(best_feature, best_threshold, left, right)


  def _most_common_label(self, y):
    counter = Counter(y)
    most_common = counter.most_common(1)[0][0]
    return most_common


  def _best_split(self, X, y, n_features):
    
    best_gain = -1
    split_idx, split_threshold = None, None

    for feature_idx in range(n_features):
      X_column = X[:, feature_idx]
      thresholds = np.unique(X_column)

      for threshold in thresholds:
        gain = self._information_gain(y, X_column, threshold)

        if gain > best_gain:
          best_gain = gain
          split_idx = feature_idx
          split_threshold = threshold

    return split_idx, split_threshold


  def _split(self, X_column, threshold): 
    
    left_idx = np.where(X_column < threshold)
    right_idx = np.where(X_column >= threshold)

    return left_idx[0], right_idx[0]


  def _information_gain(self, y, X_column, threshold):
    parent_entropy = self._entropy(y)

    left_idx, right_idx = self._split(X_column, threshold)
    if len(left_idx) == 0 or len(right_idx) == 0:
      return 0
    n, n_left, n_right = len(y), len(left_idx), len(right_idx)
    ent_left, ent_right = self._entropy(y[left_idx]), self._entropy(y[right_idx])

    child_entropy = (n_left/n) * ent_left + (n_right/n) * ent_right
    ig = parent_entropy - child_entropy
    return ig


  def _entropy(self, y):
    hist = np.bincount(y)
    ps = hist / len(y)
    return -np.sum([p * np.log(p) for p in ps if p > 0])  


  def predict(self, X):
    return np.array([self._traverse_tree(x, self.root) for x in X])


  def _traverse_tree(self, x, node):
    if node.is_leaf_node():
      return node.value

    if x[node.feature] < node.threshold:
      return self._traverse_tree(x, node.left)
    return self._traverse_tree(x, node.right)


  def _prune_tree(self, node, X_val, y_val):
    """Post-prune the tree using a validation set."""
    if node.is_leaf_node():
      return node

    # traverse 
    node.left = self._prune_tree(node.left, X_val, y_val)
    node.right = self._prune_tree(node.right, X_val, y_val)

    if node.left.is_leaf_node() and node.right.is_leaf_node():

      # split
      left_idx = np.where(X_val[:, node.feature] < node.threshold)[0]
      right_idx = np.where(X_val[:, node.feature] >= node.threshold)[0]

      #accuracy before split
      y_subset = np.concatenate([y_val[left_idx], y_val[right_idx]])
      leaf_value = self._most_common_label(y_subset)
      accuracy_before = np.sum(y_subset == leaf_value) / len(y_subset)

      #accuracy after split
      accuracy_after = np.sum(y_val[left_idx] == node.left.value) / len(y_subset)
      accuracy_after += np.sum(y_val[right_idx] == node.right.value) / len(y_subset)

      #prune
      if accuracy_after >= accuracy_before:
        return Node(value=leaf_value)

    return node

        


      


  
  
  