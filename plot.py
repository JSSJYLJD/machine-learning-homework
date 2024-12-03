import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from matplotlib.ticker import FuncFormatter

# 读取CSV文件
df = pd.read_csv('./data/mobile_price/train.csv')  # 请替换为你的实际CSV文件路径

# 设置科研绘图风格
sns.set(style="whitegrid", context="talk", palette="muted")

# 获取所有列名，排除最后一列（标签列）
columns = df.columns[:-1]  # 排除最后一列（即价格区间）

# 限制只绘制前20个变量
columns_to_plot = columns[:20]

# 设置适合竖屏的图形尺寸
fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(12, 20))  # 调整为竖屏显示比例

# 定义统一的颜色
plot_color = 'steelblue'  # 设置统一颜色

# 自定义格式化函数
def scientific_notation(x, pos):
    if x > 100:
        return f'{x:.1e}'  # 转换为科学计数法（保留1位小数）
    else:
        return f'{x:.0f}'  # 其他数值保留整数部分

# 展示每一列的条形图或直方图
for i, col in enumerate(columns_to_plot):
    ax = axes[i // 4, i % 4]  # 确定位置
    sns.histplot(df[col], ax=ax, kde=False, bins=20, color=plot_color)  # 使用 histplot 绘制
    ax.set_title(f'{col} Distribution', fontsize=14)  # 设置标题
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center', fontsize=10)  # 设置横坐标标签不倾斜，且字体较小
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)  # 设置Y轴标签字体大小

    # 设置横坐标使用自定义的科学计数法格式
    ax.xaxis.set_major_formatter(FuncFormatter(scientific_notation))  # 设置格式化函数

# 删除多余的子图
for j in range(i + 1, 5 * 4):  # 删除未使用的子图
    fig.delaxes(axes[j // 4, j % 4])

# 调整子图之间的间距
fig.subplots_adjust(hspace=0.3, wspace=0.2)  # 调整垂直和水平间距，避免重叠

# 调整布局
plt.tight_layout()
plt.show()
