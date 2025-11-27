import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report # Evaluation metrics

# Dataset
iris = load_iris()

# Printing the independent and dependent variables
print("Feature Names: ", iris.feature_names)    
print("Target Names: ", iris.target_names)

X = iris.data
y = iris.target

# Train & test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Model Training
"""
'max_iter' is set to 200 to ensure convergence for the logistic regression model.
Why do we use it?
We use max_iter to specify the maximum number of iterations the algorithm will run for during the optimization process. 
If the algorithm does not converge within this number of iterations, it will stop and return the best solution found so far. 
This is particularly useful for ensuring that the training process does not run indefinitely, 
especially when dealing with complex datasets or models that may require more iterations to converge.

What is "convergence" in this context?
Convergence in this context refers to the point at which the optimization algorithm used in logistic regression 
has sufficiently minimized the loss function, meaning that further iterations will not result in significant changes to the model parameters. 
When the model converges, it indicates that the algorithm has found a stable solution for the logistic regression coefficients that best fit the training data.

How to choose the right value for max_iter?
Choosing the right value for max_iter depends on several factors, including the complexity of the dataset, the model being used, and the desired accuracy. 
A common approach is to start with a default value (like 100 or 200) and monitor the convergence behavior. 
If the model frequently fails to converge, you may need to increase max_iter. 
Conversely, if the model converges quickly, you might be able to reduce it to save computational resources.
"""
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy: ", accuracy_score(y_test, y_pred)) # Tells how many predictions were correct
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred)) # Tells how many predictions were correct for each class
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names)) # Precision, Recall, F1-Score for each class

sample = X_test[0].reshape(1, -1) # Reshaping to make it 2D array as expected by the model, [0] is used to get the first sample from X_test
print(sample)
print(model.predict(sample))
print("\nSample Prediction: ", iris.target_names[model.predict(sample)[0]]) # [0] is used to get the first element from the prediction array

