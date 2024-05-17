# filename: train_and_plot.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import shap

# 加载数据
data = pd.read_csv('data.csv')

# 分割数据
X = data.drop('C', axis=1)  # 假设'C'是目标列，其他的是特征
y = data['C']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练线性回归模型
model = LinearRegression()
model.fit(X_train, y_train)

# 使用模型进行预测
y_pred = model.predict(X_test)

# 使用SHAP解释模型
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# 绘制小提琴图
shap.summary_plot(shap_values, X_test, plot_type="violin")

# 注意：确保你的环境能够显示图形，或者保存图形到文件中，例如：
# shap.summary_plot(shap_values, X_test, plot_type="violin", show=False)
# plt.savefig('shap_violin_plot.png')