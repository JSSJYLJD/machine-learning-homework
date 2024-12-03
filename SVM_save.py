import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 1. Load the dataset
data = pd.read_csv('./data/mobile_price/train.csv')  # Update with the actual path of your training dataset

# 2. Feature selection and target variable
X = data.iloc[:, :-1].values  # All columns except the last one are features
y = data.iloc[:, -1].values   # The last column is the target (price_range)

# 3. Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Build the Support Vector Machine Classifier
model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)  # Use RBF kernel
model.fit(X_train, y_train)

# 5. Test the model
y_pred = model.predict(X_test)

# 6. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Price Range 0', 'Price Range 1', 'Price Range 2', 'Price Range 3']))

# 7. Visualize the confusion matrix
fig, ax = plt.subplots(figsize=(6, 6))  # Set the figure size
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, display_labels=['Range 0', 'Range 1', 'Range 2', 'Range 3'], ax=ax
)

# Show the plot
plt.title("Confusion Matrix")
plt.show()

# ---------------------- Predict on test.csv ----------------------

# 8. Load the test dataset (with 20 variables)
test_data = pd.read_csv('./data/mobile_price/test.csv')  # Update with the actual path of your test dataset

# 9. Make predictions on the test dataset
X_test_new = test_data.iloc[:, 1:].values  # Assuming that the features in test.csv are all the variables
y_pred_new = model.predict(X_test_new)  # Predict price range for the test data

# 10. Save the predictions to a new CSV file
test_data['Predicted_Price_Range'] = y_pred_new  # Add predictions to the original test data
test_data.to_csv('./data/mobile_price/test_predictions.csv', index=False)  # Save the result to a new CSV file

print("Predictions saved to 'test_predictions.csv'.")
