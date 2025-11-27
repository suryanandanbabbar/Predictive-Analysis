import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer

# Dataset
data = load_diabetes()
X = data.data[:, 2].reshape(-1, 1)  # BMI Feature
# Disease Progression ("data.target" means the target variable(default from sklearn) in the dataset)
y = data.target

# Feature Scaling
scaler = StandardScaler()
x_scaled = scaler.fit_transform(X)

# Polynomial Transformation
poly = PolynomialFeatures(degree=2)
x_poly = poly.fit_transform(x_scaled)

# Train & test split
X_train, X_test, y_train, y_test = train_test_split(
    x_poly, y, test_size=0.2, random_state=0)

# Model Training
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Plotting the result
plt.scatter(x_scaled, y, color='blue', alpha=0.5)
sorted_X = np.sort(x_scaled, axis=0)  # Sorting for a better line plot
plt.plot(sorted_X, model.predict(poly.transform(sorted_X)), color='red')
plt.xlabel("BMI (Standardized)")
plt.ylabel("Disease Progression")
plt.title("Polynomial Regression (Degree 2)")
plt.show()
