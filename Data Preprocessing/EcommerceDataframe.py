import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression

df=pd.read_csv("ecommerce_customers_unit1 (1).csv",parse_dates=["signup_date","last_purchase_date"])

#The information code
print(df.head())
print("\nBasic Stats (numeric):")
print(df.describe()) # Basic stats for numerical columns like mean, std, min, max, percentiles

#Shapes and Data Types
print("Shape:",df.shape) # Shape is used to get the number of rows and columns
print("\nData Types:")
print(df.dtypes)

#Missing Values

#First the calculate the number of missing values
missing_counts=df.isna().sum().sort_values(ascending=False)
print(missing_counts)

#Plot no of missing values
plt.figure(figsize=(8,4))
missing_counts.plot(kind="bar")
plt.title("Missing values per coulmn")
plt.xlabel("Columns")
plt.ylabel("Counts")
plt.tight_layout()
plt.show()

#Dulpicate Detection and Removal
before=df.shape[0]
df=df.drop_duplicates()
after=df.shape[0]
print(f"Removed {before-after} duplicate rows. New Shape: {df.shape}")

#Handling Missing Values
#Fill Numerical data with median and categorical data with mode
num_cols=df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols=df.select_dtypes(include=["object"]).columns.tolist()

for c in num_cols:
    if df[c].isna().any():
        med=df[c].median()
        df[c].fillna(med, inplace=True)

for c in cat_cols:
    if df[c].isna().any():
        mode_val=df[c].mode(dropna=True)
        if len(mode_val)>0:
            df[c].fillna(mode_val[0],inplace=True)
        else:
            df[c].fillna("Unknown",inplace=True)
print(df.isna().sum().sort_values(ascending=False))

#Outlier Exploration (IQR method on total_spent)
q1=df["total_spent"].quantile(0.25)
q3=df["total_spent"].quantile(0.75)
iqr=q3-q1
lower=q1-1.5*iqr
upper=q3+1.5*iqr
print("IQR bounds:",lower,upper)
plt.figure(figsize=(6,4))
plt.boxplot(df["total_spent"].dropna(),vert=True)
plt.title("Boxplot: total_spent")
plt.ylabel("total_spent")
plt.tight_layout()
plt.show()

#Simple Feature Engineering
df["avg_order_value"]=np.where(df["num_orders"]>0, df["total_spent"]/df["num_orders"],0).round(2)
def recency_bucket(d):
    if d<=30: return "0-30"
    if d<=90: return "31-90"
    if d<=180: return "91-180"
    return "180+"

df["recency_bucket"]=df["recency_days"].apply(recency_bucket)
print(df[["num_orders","total_spent","avg_order_value","recency_days","recency_bucket"]].head())

#Encode Categorical Variables
#we use one hot encoding with pandas.get_dumies
categorical_cols=["gender","country","device_type","preferred_category","recency_bucket"]
df_enc=pd.get_dummies(df,columns=categorical_cols,drop_first=True) # drop_first to avoid dummy variable trap
print(df_enc.head())

#Scale Numerical Features
#we standardize selected numeric columns using z score scailing

from sklearn.preprocessing import StandardScaler

scale_cols=["age","num_orders","total_spent","avg_order_value","recency_days"]
scaler=StandardScaler()
df_scaled=df_enc.copy()
df_scaled[[f"(c)_z"for c in scale_cols]]=scaler.fit_transform(df_scaled[scale_cols])
df_scaled[[*scale_cols, *[f"(c)_z" for c in scale_cols]]].head()

#Visual Check
plt.figure(figsize=(6,4))
plt.hist(df_enc["total_spent"].dropna(),bins=30)
plt.title("Before Scaling: total_spent")
plt.xlabel("total_spent")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


plt.figure(figsize=(6,4))
plt.hist(df_scaled["total_spent"].dropna(),bins=30)
plt.title("After Scaling: total_spent")
plt.xlabel("total_spent_z")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
        
