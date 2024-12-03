import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. 数据加载和预处理
# 假设文件名为 iris.csv
column_names = ["Sepal_Length", "Sepal_Width", "Petal_Length", "Petal_Width", "Species"]
data = pd.read_csv('./data/iris.csv', header=0, names=column_names)

# 检查数据是否正确加载
print(data.head())  # 检查前5行数据

# 将类别（标签）由字符串转换为数字
data['Species'] = data['Species'].astype('category').cat.codes

# 特征和标签分离
X = data.iloc[:, :-1].values  # 特征列
y = data['Species'].values  # 标签列

# 2. 数据划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# 3. 模型训练
model = LogisticRegression(max_iter=100)
model.fit(X_train, y_train)

# 4. 模型测试
y_pred = model.predict(X_test)

# 计算准确度
accuracy = accuracy_score(y_test, y_pred)
print(f"模型准确度：{accuracy:.2f}")

# 输出分类报告
print("分类报告：")
print(classification_report(y_test, y_pred, target_names=['setosa', 'versicolour', 'virginica']))

# 可选：将预测值与真实值对比显示
results = pd.DataFrame({'真实值': y_test, '预测值': y_pred})
print(results)
