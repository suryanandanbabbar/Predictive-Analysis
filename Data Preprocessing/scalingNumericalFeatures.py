from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = {
    "Size": [800, 900, 1200, 1500, 1800],
    "No of Rooms": [1, 2, 3, 4, 5],
    "Price": [40, 50, 55, 65, 90],
    "num_orders": [2,3,1,5,4],
    "total_spent": [400,600,200,800,1000],
    "recency_days": [10,20,5,15,8],
    "age": [25,35,45,30,40],
    "avg_order_value": [200, 300, 200, 160,250],
    "total_spent": [400,600,200,800,1000]
}

df_enc = pd.DataFrame(data)

scale_cols = ["age", "num_orders", "total_spent",
              "avg_order_value", "recency_days"]

scaler = StandardScaler() # Z-Score normalization
df_scaled = df_enc.copy()
df_scaled[[f"{c}_z" for c in scale_cols]
          ] = scaler.fit_transform(df_scaled[scale_cols])

print(df_scaled[[*scale_cols, *[f"{c}_z" for c in scale_cols]]].head())

plt.figure(figsize = (6, 4))
plt.hist(df_enc["total_spent"], bins = 30)
plt.title("Before Scaling: total_spent")
plt.xlabel("total_spent")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

plt.figure(figsize = (6, 4))
plt.hist(df_scaled["total_spent"], bins = 30)
plt.title("After Scaling: total_spent")
plt.xlabel("total_spent_z")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()