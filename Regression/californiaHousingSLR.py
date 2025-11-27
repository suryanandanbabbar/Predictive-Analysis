"""
              data
                |
        _________________
       |                 |
    Income(X)       House Value(y)
    (MedInc)        (MedHouseVal)
       |                 |
       |                 |
 Feature Scaling       y_train, y_test
    (x_scaled)
       |
       |
    X_train, X_test
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler  # For feature scaling
from sklearn.impute import SimpleImputer  # For handling missing values

# Dependent Variable : Median House Value (MedHouseVal)
# Independent Variable : Median Income (MedInc)

# Loading the dataset
# 'as_frame=True' returns a pandas DataFrame
data = fetch_california_housing(as_frame=True)
# Using two columns for Simple Linear Regression
df = data.frame[['MedInc', 'MedHouseVal']]

# Introducing some missing values for preprocessing demonstration
df.loc[::50, 'MedInc'] = np.nan  # Every 50th value in 'MedInc' is set to NaN

# Handling missing values using SimpleImputer
# Since there are no outliers, using 'mean'
imputer = SimpleImputer(strategy='mean')
df['MedInc'] = imputer.fit_transform(df[['MedInc']])

# Feature and Target
X = df[['MedInc']].values
y = df['MedHouseVal'].values

# Feature Scaling using StandardScaler (Z-Score Normalization)
scaler = StandardScaler()
x_scaled = scaler.fit_transform(X)

# Train & test split
X_train, X_test, y_train, y_test = train_test_split(
    x_scaled, y, test_size=0.2, random_state=42)
"""
'42' in random_state will ensure that every time we run the code, we get the same split.
Why '0' is not used here?
Using '0' is perfectly fine, but '42' is often used as a reference to "The Hitchhiker's Guide to the Galaxy" where 42 is the "Answer to the Ultimate Question of Life, the Universe and Everything."
It's just a fun convention among programmers and data scientists.
"""

# Model Training
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Plotting the results
# 'alpha' for transparency (0 - 1)
plt.scatter(X_test, y_test, color='blue', alpha=0.5)
plt.plot(X_test, y_pred, color='red')
plt.xlabel('Median Income (scaled)')
plt.ylabel('Median House Value')
plt.title('Simple Linear Regression')
plt.show()
