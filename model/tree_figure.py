import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Load the data
data = pd.read_excel('modified_data.xlsx')  # Replace with your actual file path

# Prepare the data
X = data.drop('y', axis=1)
y = data['y'].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train_scaled, y_train)

# Plot and save the decision tree
plt.figure(figsize=(20, 10))
plot_tree(dt_classifier, filled=True, feature_names=X.columns, class_names=['0', '1'])
plt.savefig('decision_tree_path.png')  # Replace with your desired output path

# Extract feature importances
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': dt_classifier.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Plot and save the feature importances
plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importances)
plt.title('Feature Importances')
plt.tight_layout()
plt.savefig('feature_importances.png')  # Replace with your desired output path

# Save the feature importances to Excel
feature_importances.to_excel('feature_importances.xlsx', index=False)  # Replace with your desired output path
