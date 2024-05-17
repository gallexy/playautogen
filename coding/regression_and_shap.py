# filename: regression_and_shap.py
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import shap

# 假设 'data.csv' 与代码文件在同一目录下
data = pd.read_csv("data.csv")

# 将 'C' 列设为目标变量，其余列为特征
X = data.drop('C', axis=1)
y = data['C']

# 将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建线性回归模型并训练
model = LinearRegression()
model.fit(X_train, y_train)

# 使用 shap 解释模型
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# 绘制小提琴图
shap.summary_plot(shap_values, X_test, plot_type="violin")