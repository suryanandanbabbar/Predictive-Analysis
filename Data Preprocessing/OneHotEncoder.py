# This is an example of One Hot Encoding using OneHotEncoder from sklearn
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Data
data = pd.read_csv("../ecommerce_customers_unit1 (1).csv")
df = pd.DataFrame(data)

# Selecting categorical columns for encoding
categoricalCols = ["gender", "country", "device_type", "preferred_category"]
X = df[categoricalCols]

# Initializing OneHotEncoder
# drop='first' to avoid dummy variable trap and sparse_output=False to
# return a numpy array instead of a sparse matrix
encoder = OneHotEncoder(drop='first', sparse_output=False) 
X_encoded = encoder.fit_transform(X)

# Converting the encoded array back to a DataFrame for better readability
encodedCols = encoder.get_feature_names_out(categoricalCols)
df_encoded = pd.DataFrame(X_encoded, columns=encodedCols)
print(df_encoded.head())