import matplotlib.pyplot as plt
import numpy as np

# 方法 & 颜色
methods = ["MoST-LLM", "W/O MoST-MAE", "W/O Even-MoE", "W/O Temporal", "W/O Spatial"]
colors = ["#6a5acd", "#66c2a5", "#ffd92f", "#e41a1c", "#4daf4a"]

# 数据
data = {
    "Bike-In":  {"MAE": [2.67, 2.79, 2.71, 2.69, 2.70], "RMSE": [4.75, 4.89, 4.82, 4.83, 4.82]},
    "Bike-Out": {"MAE": [2.79, 2.90, 2.82, 2.83, 2.82], "RMSE": [5.05, 5.23, 5.10, 5.14, 5.16]},
    "Taxi-In":  {"MAE": [8.80, 9.65, 9.03, 9.10, 9.15], "RMSE": [18.74, 20.13, 19.33, 19.23, 19.42]},
    "Taxi-Out": {"MAE": [9.41, 10.10, 9.61, 9.56, 9.60], "RMSE": [18.34, 19.26, 18.50, 18.55, 19.60]},
}


def plot_metric(metric):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    scenarios = ["Bike-In", "Bike-Out", "Taxi-In", "Taxi-Out"]

    for ax, scenario in zip(axes.flat, scenarios):
        values = np.array(data[scenario][metric])
        ax.bar(methods, values, color=colors, width=0.6)
        ax.set_title(scenario)
        ax.set_ylabel(metric)
        ax.set_xticklabels(methods, rotation=20, ha="right")

        # 动态设定 y 轴范围 (从最小值到最大值的 5% 外扩)
        ymin, ymax = values.min(), values.max()
        margin = (ymax - ymin) * 0.1  # 上下留白 10%
        ax.set_ylim(ymin - margin, ymax + margin)

        # 给柱子加数值
        for i, v in enumerate(values):
            ax.text(i, v + margin*0.05, f"{v:.2f}", ha='center', fontsize=8)

    # 图例
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors]
    fig.legend(handles, methods, bbox_to_anchor=(0.5, 0), loc="lower center", ncol=len(methods))
    fig.suptitle(f"Ablation Study - {metric}", fontsize=14)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()


# 分别画 RMSE 和 MAE
plot_metric("RMSE")
plot_metric("MAE")