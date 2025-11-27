# This code demonstrates how to encode categorical variables using one-hot encoding
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Loading a sample ecommerce dataset
# 'parse_dates' is used to parse date columns so that they are recognized as datetime objects and not as strings
data = pd.read_csv("ecommerce_customers_unit1 (1).csv", parse_dates=["signup_date", "last_purchase_date"])
df = pd.DataFrame(data)

# Check for outliers using IQR method
Q1 = df['total_spent'].quantile(0.25)
Q3 = df['total_spent'].quantile(0.75)
IQR = Q3 - Q1
lowerBound = Q1 - 1.5 * IQR
upperBound = Q3 + 1.5 * IQR
print(f"Lower Bound: {lowerBound}, Upper Bound: {upperBound}")
"""
Lower Bound: -4482.915, Upper Bound: 17926.5650
This indicates that any 'total_spent' value below -4482.915 or above 17926.5650 is considered an outlier.
Next, we can visualize these outliers using a boxplot.
Then, if there are outliers, we can handle them by capping or removing them.
"""
# Plotting outliers for 'total_spent'
# plt.figure(figsize=(8, 4))
# plt.boxplot(df["total_spent"].dropna(), vert=True) # dropna to avoid errors of NaN values in boxplot
# plt.title("Boxplot for total_spent")
# plt.xlabel("total_spent")
# plt.tight_layout()
# plt.show()

# Removing outliers
print(f"Data shape before removing outliers: {df.shape}")
df = df[(df['total_spent'] >= lowerBound) & (df['total_spent'] <= upperBound)]
print(f"Data shape after removing outliers: {df.shape}")

# Checking for Missing values
missingValues = df.isnull().sum()
missingValues.plot(kind='bar', figsize=(8, 4))
# plt.title("Missing Values per Column")
# plt.xlabel("Columns")
# plt.ylabel("Counts")
# plt.tight_layout()
# plt.show()

# Handling Missing Values using SimpleImputer
# Since, outliers are present, we use median strategy for numerical columns
imputer = SimpleImputer(strategy='median')
numCols = df.select_dtypes(include = [np.number]).columns.tolist() # Get numerical columns and convert to list
df[numCols] = imputer.fit_transform(df[numCols])
# print("Missing values after imputation:")
# print(df.isnull().sum())

# Encoding Categorical Variables
# we use one hot encoding with pandas.get_dummies
categoricalCols = ["gender", "country", "device_type", "preferred_category"]
dfEnc = pd.get_dummies(df, columns = categoricalCols, drop_first=True) # drop_first to avoid dummy variable trap
#encoder = OneHotEncoder(sparse_output=False)
#X = df[categoricalCols]
#X_encoded = encoder.fit_transform(X)

# print(dfEnc.head())

# Feature Scaling
scaler = StandardScaler()
numericalFeatures = ["age", "num_orders", "total_spent", "avg_order_value", "recency_days"]
dfScaled = dfEnc.copy()
dfScaled[[f"{col}_z" for col in numericalFeatures]] = scaler.fit_transform(dfScaled[numericalFeatures])
# print(dfScaled.head())  