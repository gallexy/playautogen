# filename: elastic_net.py
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("data.csv")  # 读取数据

X = data.drop('C', axis=1)
y = data['C']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = ElasticNet(alpha=0.1, l1_ratio=0.5)  # 创建弹性网络回归模型
model.fit(X_train, y_train)  # 训练模型

y_pred = model.predict(X_test)  # 预测

mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R-squared:", r2)