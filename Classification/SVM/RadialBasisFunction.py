import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Load Dataset
data = datasets.load_iris()
X = data.data[:, :2]  # 2D plane
y = data.target
class_names = data.target_names

print(X[:5])
print(y[:5])
print(class_names)

# Train-Test Split
"""
stratifiy=y ensures that the class distribution in the train and test sets
is similar to that in the original dataset. This is particularly important
when dealing with imbalanced datasets, as it helps to maintain the
proportion of each class in both subsets, leading to more reliable and 
representative model evaluation.
"""
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

# Train SVM
# rbf stands for Radial Basis Function
# C=1 is the regularization parameter used to control the trade-off between achieving a low
# training error and a low testing error that is, generalization.
# gamma='scale' is a parameter for non-linear hyperplanes. The higher the gamma value,
# the more it tries to fit the training data set.
model = SVC(kernel='rbf', C=0.1, gamma='scale')
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Classification Report:\n", classification_report(y_test, y_pred))

# Visualisation
plt.figure(figsize=(7, 5))
for i, label in enumerate(class_names):
    plt.scatter(X[y == i, 0], X[y == i, 1], label=label)

plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Iris Dataset (2 features)")
plt.legend()
plt.grid(True)
plt.show()