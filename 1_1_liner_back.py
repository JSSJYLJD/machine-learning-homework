import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import matplotlib
# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 或者使用其他支持中文的字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# 数据
area = np.array([50, 70, 88, 69, 100, 120]).reshape(-1, 1)  # 房屋面积（二维数组）
price = np.array([47, 72, 80, 77, 110, 100])  # 房屋价格

# 创建线性回归模型并训练
model = LinearRegression()
model.fit(area, price)

# 使用模型预测面积为130的房屋价格
predicted_price = model.predict(np.array([[130]]))
print(f"预测面积为130平方米的房屋价格为：{predicted_price[0]:.2f}万元")

# 可视化
plt.scatter(area, price, color='blue', label='真实数据')
plt.plot(area, model.predict(area), color='red', label='回归线')
plt.scatter([130], predicted_price, color='green', label='预测点')
plt.xlabel('房屋面积 (平方米)')
plt.ylabel('房屋价格 (万元)')
plt.legend()
plt.show()
