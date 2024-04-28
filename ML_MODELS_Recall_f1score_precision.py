import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the data from the backend
# Replace 'your_data.csv' with the actual file containing your data
data = pd.read_csv('sleep.csv')

# Assuming 'Person ID' is not useful for prediction, dropping it
data = data.drop('Person ID', axis=1)

# Handle categorical data (if any)
le = LabelEncoder()
for column in ['Gender', 'Occupation', 'BMI Category']:
  data[column] = le.fit_transform(data[column])

# Split the 'Blood Pressure' column into 'Systolic' and 'Diastolic'
data[['Systolic BP', 'Diastolic BP']] = data['Blood Pressure'].str.split('/', expand=True)

# Convert the new columns to numeric values
data[['Systolic BP', 'Diastolic BP']] = data[['Systolic BP', 'Diastolic BP']].apply(pd.to_numeric)

# Drop the original 'Blood Pressure' column
data = data.drop('Blood Pressure', axis=1)

# Define features (X) and target variable (y)
features = ['Age', 'Sleep Duration', 'Quality of Sleep', 'BMI Category', 'Systolic BP', 'Diastolic BP', 'Heart Rate', 'Daily Steps']
X = data[features]
y = data['Sleep Disorder']

# Handle missing values using SimpleImputer for features
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Handle missing values in the target variable
y_imputer = SimpleImputer(strategy='most_frequent')  # You can use a different strategy if needed
y = y_imputer.fit_transform(y.values.reshape(-1, 1)).ravel()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a decision tree classifier
clf_decision_tree = DecisionTreeClassifier()

# Create a random forest classifier
clf_random_forest = RandomForestClassifier(n_estimators=100)  # You can adjust this parameter

# Train the decision tree and random forest models
clf_decision_tree.fit(X_train, y_train)
clf_random_forest.fit(X_train, y_train)

# Create a KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)  # You can tune this parameter

# Train the KNN model
knn.fit(X_train, y_train)

# Evaluate the models on the test set
def evaluate_model(model, y_true, y_pred):
  accuracy = accuracy_score(y_true, y_pred)

  # Option 1: Calculate precision for each class (average=None)
  precision = precision_score(y_true, y_pred, average=None)
  print(f"Precision (per class): {precision}")

  # Option 2: Micro-averaging (average='micro')
  precision_micro = precision_score(y_true, y_pred, average='micro')
  print(f"Precision (micro-averaged): {precision_micro:.4f}")

  # Option 3: Macro-averaging (average='macro')
  precision_macro = precision_score(y_true, y_pred, average='macro')
  print(f"Precision (macro-averaged): {precision_macro:.4f}")

  # Option 4: Weighted averaging (average='weighted')
  precision_weighted = precision_score(y_true, y_pred, average='weighted')
  print(f"Precision (weighted-averaged): {precision_weighted:.4f}")

  recall = recall_score(y_true, y_pred, average='macro')
  f1 = f1_score(y_true, y_pred, average='macro')
  print(f"Recall: {recall:.4f}")
  print(f"F1-score: {f1:.4f}")


# Evaluate Decision Tree
print("Evaluation for Decision Tree:")
y_pred_decision_tree = clf_decision_tree.predict(X_test)
evaluate_model(clf_decision_tree, y_test, y_pred_decision_tree)

# Evaluate Random Forest
print("\nEvaluation for Random Forest:")
y_pred_random_forest = clf_random_forest.predict(X_test)
evaluate_model(clf_random_forest, y_test, y_pred_random_forest)

# Evaluate KNN
print("\nEvaluation for KNN:")
y_pred_knn = knn.predict(X_test)
evaluate_model(knn, y_test, y_pred_knn)
