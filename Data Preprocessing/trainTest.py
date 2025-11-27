import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 5, 4, 5])

# Splitting the dataset into training and testing sets
# Ideal percentage is 70-80% training and 20-30% testing
# 'test_size' parameter is used to specify the percentage of data to be used for testing (0.2 -> 20%)
# 'random_state' parameter is used to ensure reproducibility of results (same split every time the code is run)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

print(f"Intercept: {model.intercept_: .2f}")
print(f"Coefficient: {model.coef_[0]: .2f}")

# Plotting the results
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, model.predict(X), color='red', linewidth = 2, label = "Regression Line")
plt.xlabel("Independent Variable (X)")
plt.ylabel("Dependent Variable (y)")
plt.title("Linear Regression Fit")
plt.legend()  
plt.show()