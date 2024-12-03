import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the dataset
data = pd.read_csv('./data/mobile_price/train.csv')  # Update with the actual path of your dataset

# 2. Feature selection and target variable
X = data.iloc[:, :-1].values  # All columns except the last one are features
y = data.iloc[:, -1].values   # The last column is the target (price_range)

# 3. Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Build the Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Test the model
y_pred_continuous = model.predict(X_test)  # Predict continuous values
y_pred = np.round(y_pred_continuous).astype(int)  # Round predictions to nearest integer (class labels)

# 6. Ensure predictions fall within valid range
y_pred = np.clip(y_pred, 0, 3)  # Limit predictions to the range of [0, 3]

# 7. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Price Range 0', 'Price Range 1', 'Price Range 2', 'Price Range 3']))

# 8. Visualize the confusion matrix
fig, ax = plt.subplots(figsize=(6, 6))  # Set the figure size
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, display_labels=['Range 0', 'Range 1', 'Range 2', 'Range 3'], ax=ax
)

# Show the plot
plt.title("Confusion Matrix")
plt.show()
