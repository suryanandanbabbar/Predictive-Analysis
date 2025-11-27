import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# In this, implementing Multiple Linear Regression (MLR)
# Dependent Variable : Median House Value (MedHouseVal)
# Independent Variables : All other features in the dataset

# Loading the dataset
data = fetch_california_housing(as_frame=True)
df = data.frame[['MedInc', 'HouseAge', 'AveRooms', 'AveOccup', 'MedHouseVal']]

# Introducing missing values
df.loc[::100, 'HouseAge'] = np.nan

# Handling missing values
imputer = SimpleImputer(strategy='mean')
df[['HouseAge']] = imputer.fit_transform(df[['HouseAge']])

# Feature and Target
X = df[['MedInc', 'HouseAge', 'AveRooms', 'AveOccup']]
y = df['MedHouseVal']

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train & test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=0)

# Model Training
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Plotting Actual vs Predicted
"""
In MLR, since we have multiple features, we cannot plot a simple 2D graph.
Instead, we will plot Actual vs Predicted values to see how well our model performed.
Why are we taking "y_test" and "y_pred" for plotting instead of "X_test" and "y_pred"?
In SLR, we take "X_test" because we have a single independent variable, so we can plot it against the predicted values.
In MLR, we have multiple independent variables, making it impossible to represent them all in a 2D plot.
Hence, we plot the actual target values (y_test) against the predicted values (y_pred) to visualize the model's performance.
"""

plt.scatter(y_test, y_pred, color='green', alpha=0.5)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Multiple Linear Regression")
plt.show()
