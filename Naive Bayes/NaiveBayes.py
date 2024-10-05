import numpy as np

class NaiveBayes:
    
    def __init__(self):
        pass
    
    def fit(self, X,y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self.classes)
        
        self._mean = np.zeros((n_classes, n_features))
        self._var = np.zeros((n_classes, n_features))
        self._prior = np.zeros(n_classes)
        
        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self._mean[idx, :] = X_c.mean(axis=0)
            self._var[idx, :] = X_c.var(axis=0)
            # Laplace Smoothing
            self._prior[idx] = (X_c.shape[0]+1) / (n_samples+n_classes) 
            
    def predict(self, X):
        preds = [self._predict(x) for x in X]
        return np.array(preds)
        
    def _predict(self, x):
        posteriors = []
        for idx, c in enumerate(self.classes):
            posterior = np.log(self._prior[idx])
            posterior += np.sum(np.log(self._pdf(idx, x)))
            
            
            
    def _pdf(idx, x):
        mean_class = self._mean[idx]
        var_class = self._var[idx]
        num = np.exp(- (x-mean_class)**2 / (2* var_class))
        den = np.sqrt(2 * np.pi * var_class)
        
        
        