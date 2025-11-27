import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Dataset
iris = load_iris()
X = iris.data   # features (sepal length, sepal width, petal length, petal width)
y = iris.target # target (species type)

# Train & Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()

"""
Why .fit_trainsform() on training data and only .transform() on test data?
Because we want to scale the test data based on the parameters (mean and std) learned from the training data.
This ensures that the model is not biased by information from the test set during training.

Explain further with an example:
If we fit the scaler on the test data, it might have a different mean and standard deviation
compared to the training data. This would lead to inconsistent scaling between the training and test sets,
potentially degrading model performance. By fitting only on the training data, we ensure that both sets
are scaled in the same way, maintaining the integrity of the evaluation process.

"""
X_train = scaler.fit_transform(X_train) 
X_test = scaler.transform(X_test)

# Training KNN Model
# By default, distance matrix is "Euclidean Distance"
k = 5 # default for iris
knn = KNeighborsClassifier(n_neighbors=k) 
knn.fit(X_train, y_train) # Doesn't compute but stores the training data

# Predict
y_pred = knn.predict(X_test) # Now it will compute
print("Predicted labels: ", y_pred)
# Now converting labels to values
y_pred_values = iris.target_names[y_pred]
print("Predicted species: ", y_pred_values)
# Evaluate
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

