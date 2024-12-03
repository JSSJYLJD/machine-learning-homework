import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt  # 导入matplotlib以显示图形

# 1. Load the dataset
data = pd.read_csv('./data/mobile_price/train.csv')  # Update with the actual path of your dataset

# 2. Feature selection and target variable
X = data.iloc[:, :-1].values  # All columns except the last one are features
y = data.iloc[:, -1].values   # The last column is the target (price_range)

# 3. Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Standardize the data (important for neural networks)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Build the Neural Network Classifier
model = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=500, random_state=42)
model.fit(X_train, y_train)

# 6. Test the model
y_pred = model.predict(X_test)

# 7. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Price Range 0', 'Price Range 1', 'Price Range 2', 'Price Range 3']))

# 8. Visualize the confusion matrix
fig, ax = plt.subplots(figsize=(6, 6))  # 设置图形大小
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, display_labels=['Range 0', 'Range 1', 'Range 2', 'Range 3'], ax=ax
)

# Show the plot
plt.title("Confusion Matrix")
plt.show()
