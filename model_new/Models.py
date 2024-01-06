import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve, confusion_matrix, \
    auc
import matplotlib.pyplot as plt

# Load the Excel file
file_path = 'data_final.xlsx'  # Replace with your actual file path
data = pd.read_excel(file_path)

# Impute missing values
# For continuous features use mean and for categorical features use mode
imputer_cont = SimpleImputer(strategy='mean')
imputer_cat = SimpleImputer(strategy='most_frequent')

# Assuming that columns are correctly identified as continuous or categorical
# You might need to adjust this part based on your actual data
for col in data.columns:
    if data[col].dtype in ['int64', 'float64']:
        data[col] = imputer_cont.fit_transform(data[[col]])
    else:
        data[col] = imputer_cat.fit_transform(data[[col]])

# Define X and y
X = data.drop('y', axis=1)
y = data['y'].astype(int)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(eval_metric='logloss'),
    "SVM": SVC(probability=True)
}

# DataFrame to hold model results
model_results = pd.DataFrame(columns=["Model", "Accuracy", "Precision", "Recall", "AUC"])

# Train and evaluate models
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(
        X_test_scaled)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    auc_score = roc_auc_score(y_test, y_proba)

    # Append results to the DataFrame
    model_results = model_results._append({"Model": name,
                                          "Accuracy": accuracy,
                                          "Precision": precision,
                                          "Recall": recall,
                                          "AUC": auc_score}, ignore_index=True)

# Save model results to Excel
model_results.to_excel('model_results.xlsx', index=False)

# Export confusion matrices to Excel
with pd.ExcelWriter('confusion_matrices.xlsx') as writer:
    for name, model in models.items():
        y_pred = model.predict(X_test_scaled)
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(cm, index=['Actual 0', 'Actual 1'], columns=['Predicted 0', 'Predicted 1'])
        cm_df.to_excel(writer, sheet_name=name)

# Plot ROC curves
plt.figure(figsize=(10, 8))
for name, model in models.items():
    y_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(
        X_test_scaled)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc(fpr, tpr):.2f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.savefig('roc_curves.png')
plt.show()
