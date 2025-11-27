from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Example1 illustrating data is linear
x1 = np.array([800, 1000, 1200, 1500, 1800])
y1 = np.array([45, 60, 70, 85, 90])

# Example2 illustrating data is parabolic
x = np.array([800, 1000, 1200, 1500, 1800])
y = np.array([45, 50, 55, 70, 85])

plt.scatter(x, y, color='red', label='Data Points')
plt.xlabel("Independent Variable (x)")
plt.ylabel("Dependent Variable (y)")
plt.title("Scatter Plot of x vs y")  # Parabolic relationship
# plt.show()


# Polynomial Regression

# Same as Example2
X = np.array([800, 1000, 1200, 1500, 1800]).reshape(-1, 1)
Y = np.array([45, 50, 55, 70, 85])

# Transform data to include X^2 (since the data is parabolic)
# Preparing the machine to perform the task for degree '2' (and not actually doing it), creates another column/feature for X^2
poly = PolynomialFeatures(degree=2)
x_poly = poly.fit_transform(X)

# Model
model = LinearRegression()  # No matter linear or polynomial, we use LinearRegression()
# Fitting the curved line to the data points (dependent variable: Y, independent variable: x_poly)
model.fit(x_poly, Y)

print("Intercept(b0): ", model.intercept_)
print("Coefficients (b1, b2): ", model.coef_)

# Predicting values
# New data for prediction (100 data points between 800 and 1800)
x_new = np.linspace(800, 1800, 100).reshape(-1, 1)
# Predicting the values for new data points ("poly.transform" is used to transform the new data points to include X^2)
y_pred = model.predict(poly.transform(x_new))

# Plotting the results
plt.scatter(X, Y, color='blue', label='Data Points')
plt.plot(x_new, y_pred, color='green', label='Polynomial Regression Fit')
plt.xlabel("Independent Variable (x)")
plt.ylabel("Dependent Variable (y)")
plt.title("Polynomial Regression Fit")
plt.legend()
plt.show()
