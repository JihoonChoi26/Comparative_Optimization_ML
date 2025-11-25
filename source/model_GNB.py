import numpy as np
import pandas as pd

class SimpleGaussianNB:
    '''
    Simplified Gaussian Naive Bayes Classifier
        
    Core Ideas:
    - Bayes' Theorem: P(y|x) ∝ P(y) * P(x|y)
    - Naive Assumption: Features are independent of each other
    - Gaussian Assumption: Each feature follows a normal distribution
        
    Therefore: P(x|y) = ∏ P(x_i|y) where P(x_i|y) ~ N(μ_yi, σ²_yi)
    '''
    def __init__(self):
        '''
        Initialization: Parameters to be stored after training
        '''
        self.classes = None # Class labels (e.g., [0, 1])
        self.class_priors = None # P(y): Prior probability of each class
        self.means = None # μ: Mean for each (class, feature) pair
        self.variances = None # σ²: Variance for each (class, feature) pair
        
    def fit(self, X, y):
        '''
        Train the model: Calculate mean, variance, and prior probability for each class
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix of training data
        y : array-like, shape (n_samples,)
            Target labels of training data
            
        Returns:
        --------
        self : Trained model object
        
        Training Process:
        1. Separate data by each class
        2. Calculate prior for each class: P(y=c) = (# of samples in class c) / (total # of samples)
        3. Calculate mean (μ) and variance (σ²) for each (class, feature) pair
        '''
        # Convert DataFrame to numpy array (for consistency in internal calculations)
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        # Extract unique class labels (e.g., [0, 1])
        self.classes = np.unique(y)
        n_classes = len(self.classes) # Number of classes (2 for binary)
        n_features = X.shape[1] # Number of features
        
        # Initialize arrays to store training results
        # class_priors: shape (n_classes,) - one probability per class
        # means: shape (n_classes, n_features) - mean for each class and feature
        # variances: shape (n_classes, n_features) - variance for each class and feature
        self.class_priors = np.zeros(n_classes)
        self.means = np.zeros((n_classes, n_features))
        self.variances = np.zeros((n_classes, n_features))
        
        # Calculate statistics for each class
        for i, c in enumerate(self.classes):
            # Extract only samples belonging to current class c
            # X_c: shape (n_samples_in_class_c, n_features)
            X_c = X[y == c]
            
            # 1. Calculate prior probability: P(y=c)
            # Proportion of class c among all samples
            self.class_priors[i] = len(X_c) / len(X)
            
            # 2. Calculate mean for each feature: μ_c
            # X_c.mean(axis=0): Calculate mean for each column (feature)
            # Result: shape (n_features,) - one mean value per feature
            self.means[i] = X_c.mean(axis=0)
            
            # 3. Calculate variance for each feature: σ²_c
            # X_c.var(axis=0): Calculate variance for each column (feature)
            # Result: shape (n_features,) - one variance value per feature
            self.variances[i] = X_c.var(axis=0) + 1e-9 # + 1e-9: Prevent variance from becoming 0 (avoid division by zero later)
            
        return self
        
    def _gaussian_log_likelihood(self, x, mean, var):
        '''
        Calculate log-likelihood of Gaussian distribution for a single sample
        
        Logarithm
        - Multiplying very small probability values --> risk of underflow
        - In log space, multiplication becomes addition, which is numerically stable
        
        Parameters:
        -----------
        x : array-like, shape (n_features,)
            Feature vector of a single sample
        mean : array-like, shape (n_features,)
            Mean of each feature for a specific class
        var : array-like, shape (n_features,)
            Variance of each feature for a specific class
            
        Returns:
        --------
        log_likelihood : float
            log P(x|y) = sum over all log P(x_i|y)
        '''
        # Term 1: -0.5 * Σ log(2πσ²)
        # Calculate for each feature and sum all
        term1 = -0.5 * np.sum(np.log(2 * np.pi * var))
        
        # Term 2: -0.5 * Σ (x_i - μ)² / σ²
        # For each feature, divide (observed - mean)² by variance, then sum all
        term2 = -0.5 * np.sum((x - mean)**2 / var)
        
        return term1 + term2
    
    def predict(self, X):
        '''
        Predict classes for test data
        
        For each sample x:
        1. Calculate posterior probability for each class c
        2. Select the class with highest posterior probability
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples to predict
            
        Returns:
        --------
        predictions : array, shape (n_samples,)
            Predicted class for each sample
        '''
        # Convert DataFrame to numpy array
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        predictions = []
        
        # Iterate through each sample (row) for prediction
        for x in X: # x: shape (n_features,) - single sample
            # Calculate posterior probability for each class
            posteriors = [] # Store log P(y|x) for each class
            
            # Calculate posterior probability for each class
            for i, c in enumerate(self.classes):
                # 1) Prior probability: log P(y=c)
                # self.class_priors[i] is a value between 0~1, so take log
                prior = np.log(self.class_priors[i])
                
                # 2) Likelihood: log P(x|y=c)
                # Probability that current sample x comes from distribution of class c
                # self.means[i]: Mean of each feature for class c (shape: n_features,)
                # self.variances[i]: Variance of each feature for class c (shape: n_features,)
                likelihood = self._gaussian_log_likelihood(
                    x, self.means[i], self.variances[i]
                )
                
                # 3) Posterior probability: log P(y=c|x) ∝ log P(y=c) + log P(x|y=c)
                # Log version of Bayes' Theorem
                # (Denominator P(x) is common to all classes, so can be omitted for comparison)
                posteriors.append(prior + likelihood) # e.g., list [np.float64(-77.65), np.float64(-53.45)]
            
            # 4) Select the class with highest posterior probability
            # np.argmax(posteriors): Returns index of maximum value
            # self.classes[index]: Actual class label at that index
            predictions.append(self.classes[np.argmax(posteriors)]) # -77.65 < -53.45 => index 1
        
        return np.array(predictions)
    
    def predict_proba(self, X):
        '''
        Return probability for each class
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples to predict
            
        Returns:
        --------
        probabilities : array, shape (n_samples, n_classes)
            Probability of each class for each sample
            e.g., [[0.8, 0.2], [0.3, 0.7], ...] (for binary classification)
            
        Probability Calculation Process:
        ---------------------------------
        1. Calculate log posterior probability for each class (same as predict)
        2. Convert to actual probabilities using Softmax transformation
           P(y=c|x) = exp(log P(y=c|x)) / Σ exp(log P(y=k|x))
        3. Use log-sum-exp trick for numerical stability
        '''
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        probs = []
        
        # Iterate through each sample (row)
        for x in X: # x: shape (n_features,) - single sample
            posteriors = []
            
            # Calculate log posterior probability for each class (same as predict method)
            for i in range(len(self.classes)):
                prior = np.log(self.class_priors[i])
                likelihood = self._gaussian_log_likelihood(
                    x, self.means[i], self.variances[i]
                )
                posteriors.append(prior + likelihood)
                
            # Convert posteriors to numpy array
            # shape: (n_classes,) - log P(y|x) for each class
            posteriors = np.array(posteriors) # list -> ndarray object

            # Convert to actual probabilities using softmax transformation (P(y=c|x) = exp(log P(y=c|x)) / Σ exp(log P(y=k|x)))
            # But exp() can create very large/small values causing overflow/underflow)
            # --> So using softmax transformation with log-sum-exp trick
            # 1) Subtract maximum value for stabilization: posteriors - max(posteriors)
            #    --> makes the largest value 0, and the rest negative
            #    --> mathematically equivalent: exp(a-M) / Σexp(b-M) = exp(a) / Σexp(b)
            posteriors = posteriors - np.max(posteriors)
            
            # 2) Apply exponential function
            # Now all values are ≤ 0, so no overflow
            exp_posteriors = np.exp(posteriors)
            
            # 3) Normalize
            # Divide each value by total sum to convert to probability
            # Sum of results = 1.0 (axiom of probability)
            probs.append(exp_posteriors / np.sum(exp_posteriors))
            
        # shape: (n_samples, n_classes)
        # Each row is one sample, each column is one class
        return np.array(probs)