import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_excel('data_final.xlsx')  # 替换为您的文件路径

# 分离特征和目标变量
X = data.drop('y', axis=1)
y = data['y']

# 填补缺失值
# 连续变量使用均值填补，01变量使用众数填补
imputer_cont = SimpleImputer(strategy='mean')
imputer_cat = SimpleImputer(strategy='most_frequent')

for col in X.columns:
    if X[col].dtype == 'object' or len(X[col].unique()) == 2:
        X[col] = imputer_cat.fit_transform(X[[col]])
    else:
        X[col] = imputer_cont.fit_transform(X[[col]])

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 标准化数据
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 模型初始化
models = {
    'Logistic Regression': LogisticRegression(),
    'Lasso Regularization': Lasso(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'XGBoost': XGBClassifier(),
    'SVM': SVC()
}

# 训练模型并获取特征重要性
feature_importances = pd.DataFrame(index=X.columns)

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    if hasattr(model, 'feature_importances_'):
        feature_importances[name] = model.feature_importances_
    elif hasattr(model, 'coef_'):
        feature_importances[name] = model.coef_[0]

# 保存特征重要性到Excel
feature_importances.to_excel('feature_importances.xlsx')

# 绘制决策树路径图
dt_model = models['Decision Tree']
plt.figure(figsize=(20, 10))
plot_tree(dt_model, filled=True, feature_names=X.columns, class_names=['0', '1'], max_depth=3)
plt.savefig('decision_tree.png')
plt.show()
