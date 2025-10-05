import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 创建数据框
data = {
    'Model': ['MoSSL', 'MTGNN', 'MSTFCN', 'DMSTGCN', 'STG-LLM', 'STP-PLM', 'ST-LLM', 'AGCRN', 'MoST-LLM'],
    'Parameters': [2085993, 259440, 97788, 601984, 82608192, 127127121, 82903104, 1137940, 124739972],
    'Time': [13.8, 8.1, 11.7, 10.2, 17.6, 27, 11.2, 24.1, 23],
    'MAE': [2.92, 3.62, 2.72, 3.29, 4.93, 3.04, 3.1, 4.62, 2.66]
}

df = pd.DataFrame(data)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

# 创建图表
fig, ax = plt.subplots(figsize=(12, 8))

# 归一化参数值以便更好地显示圆圈大小
# 使用对数缩放以更好地处理参数数量的巨大差异
norm_params = np.log10(df['Parameters'])
sizes = 100 * (norm_params - norm_params.min() + 1)  # 调整大小以便可视化

# 定义颜色映射
colors = plt.cm.tab10(np.linspace(0, 1, len(df)))

# 绘制散点图
scatter = ax.scatter(df['Time'], df['MAE'], s=sizes, c=colors, alpha=0.7, edgecolors='black')

# 添加模型标签
for i, model in enumerate(df['Model']):
    ax.annotate(model, (df['Time'][i], df['MAE'][i]),
                xytext=(5, 5), textcoords='offset points',
                fontsize=18, ha='left')

# 设置坐标轴标签和标题
ax.set_xlabel('Training Time (s/epoch)', fontsize=20)
ax.set_ylabel('MAE', fontsize=20)


# 设置网格
ax.grid(True, alpha=0.3)

# 设置坐标轴范围
ax.set_xlim(5, 30)
ax.set_ylim(2, 5.5)
ax.tick_params(axis='both', which='major', labelsize=20)

# 创建图例说明参数数量级
# 添加参数数量级的图例
size_legend_elements = [
    plt.scatter([], [], s=100*(np.log10(10**5)+1), c='gray', alpha=0.7, edgecolors='black'),
    plt.scatter([], [], s=100*(np.log10(10**6)+1), c='gray', alpha=0.7, edgecolors='black'),
    plt.scatter([], [], s=100*(np.log10(10**7)+1), c='gray', alpha=0.7, edgecolors='black'),
    plt.scatter([], [], s=100*(np.log10(10**8)+1), c='gray', alpha=0.7, edgecolors='black')
]

size_labels = ['10^5', '10^6', '10^7', '10^8']

legend1 = ax.legend(size_legend_elements, size_labels, title='Parameters', loc='upper right', fontsize=17, title_fontsize=18)
ax.add_artist(legend1)

# 保存图表
plt.tight_layout()
plt.savefig('model_performance_comparison.png', dpi=300, bbox_inches='tight')
plt.show()