import numpy as np
from tqdm import tqdm

class LinearRegression:
    
    def __init__(self, num_epochs=100, learning_rate=0.01):
        self.coefs_ = None
        self.intercept_ = None
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        
    def fit(self, X, y):
        
        self.coefs_ = np.random.rand(X.shape[1])
        self.intercept_ = np.random.rand()
        
        for _ in tqdm(range(self.num_epochs)):
            
            y_pred = np.matmul(X, self.coefs_) + self.intercept_
            
            gradient_intercept_per_row = np.sum(-2 * (y - y_pred))
            gradient_intercept_ = np.sum(gradient_intercept_per_row)
            
            gradient_coefs_per_row = -2 * np.matmul(y - y_pred, X)
            gradient_coefs_ = np.sum(gradient_coefs_per_row, axis=0)
            
            self.coefs_ -= self.learning_rate * gradient_coefs_
            self.intercept_ -= self.learning_rate * gradient_intercept_
            
    def predict(self, X):
        return np.matmul(X, self.coefs_) + self.intercept_
    
    