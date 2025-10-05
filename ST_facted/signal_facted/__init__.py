# import os
# import torch
# import torch.nn as nn
# import numpy as np
#
# import pandas as pd
# import matplotlib.pyplot as plt
#
# plt.figure(figsize=(14, 6))
# hours = np.arange(0,240)
#
#
#
#
#
# # Load prediction and true value data
#
#
# data_pred = np.load('pred.npy')
#
# data_true = np.load('true.npy')
#
#
#
# # Extract relevant data for the plots
# x_pig1 = data_pred[:240,0, 48,0]  # Predictions for one day
# x_pig2 = data_true[:240,0, 48,0]  # True Values for one day
#
#
#   # Predictions for one day
# #x_pig2 = data_true[0:288,0, 22]  # True Values for one day
#
#
# # Plotting the data
# plt.plot(hours, x_pig2, 'r', alpha=0.8, linewidth=1.8, label='True Value')
# plt.plot(hours, x_pig1, 'b--', alpha=0.8, linewidth=1.8, label='Reconstructed Value')
#
#
#
# plt.tick_params(axis='both', which='major', labelsize=22)  # Major ticks
# plt.tick_params(axis='both', which='minor', labelsize=22)
#
# # Set labels and their font sizes
# #plt.xlabel('Time (HH:MM)', fontsize=16)  # Set font size for x-label
# #plt.ylabel('Values', fontsize=16)  # Set font size for y-label
# #plt.title('Predicted vs True Values Over 24 Hours', fontsize=14)  # Set font size for title
#
# plt.grid()
# plt.legend(fontsize=22)  # Set font size for legend
# ax = plt.gca()  # 获取当前坐标轴
# for spine in ax.spines.values():
#     spine.set_linewidth(2.5)  # 设置边框线宽为2.5
#     spine.set_color('black')
# # Adjust layout and show the plot
# plt.tight_layout()
# plt.show()


import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 设置重构率 (0.0 到 1.0)
RECONSTRUCTION_RATE = 0.25  # 显示25%的重构值
NUM_SEGMENTS = 6  # 设置连续重构段的数量

hours = np.arange(0, 240)

# 加载预测值和真实值数据
data_pred = np.load('pred.npy')
data_true = np.load('true.npy')

# 提取数据
x_pig1 = data_pred[:240, 0, 60, 0]  # 预测值
x_pig2 = data_true[:240, 0, 60, 0]  # 真实值

# 计算每个重构段的长度
segment_length = int(len(hours) * RECONSTRUCTION_RATE / NUM_SEGMENTS)

# 创建重构值数组（初始全为NaN）
x_pig1_masked = np.full_like(x_pig1, np.nan)

# 存储所有重构段的起始和结束索引
segments = []

# 创建图表
plt.figure(figsize=(14, 6))
plt.rcParams["font.family"] = "Times New Roman"
# 绘制真实值（完整曲线）
plt.plot(hours, x_pig2, 'r', alpha=0.8, linewidth=1.8, label='True Value')

# 生成多个连续重构段
for i in range(NUM_SEGMENTS):
    # 随机选择连续重构段的起始点
    start_idx = np.random.randint(0, len(hours) - segment_length + 1)
    end_idx = start_idx + segment_length

    # 确保段之间不重叠
    while any(start_idx < seg_end and end_idx > seg_start
              for seg_start, seg_end in segments):
        start_idx = np.random.randint(0, len(hours) - segment_length + 1)
        end_idx = start_idx + segment_length

    segments.append((start_idx, end_idx))

    # 设置连续重构段的值
    x_pig1_masked[start_idx:end_idx] = x_pig1[start_idx:end_idx]

    # 绘制重构段
    plt.plot(hours[start_idx:end_idx], x_pig1_masked[start_idx:end_idx],
             'b--', alpha=0.8, linewidth=1.8)

    # 在重构段的两端添加标记点
    plt.scatter([hours[start_idx], hours[end_idx - 1]],
                [x_pig1[start_idx], x_pig1[end_idx - 1]],
                color='blue', s=80, zorder=5, edgecolor='black')

    # 添加重构段背景
    plt.axvspan(hours[start_idx], hours[end_idx - 1], alpha=0.1, color='blue')

# 添加图例
plt.plot([], [], 'b-', alpha=0.8, linewidth=1.8, label='Reconstructed Value')

# 设置图表样式
plt.tick_params(axis='both', which='major', labelsize=22)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=22, loc='best')
plt.tight_layout()

# 添加标题说明重构率和段数
ax = plt.gca()  # 获取当前坐标轴
for spine in ax.spines.values():
    spine.set_linewidth(2.5)  # 设置边框线宽为2.5
    spine.set_color('black')  # 设置边框颜色为黑色


plt.show()


