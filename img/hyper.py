# import matplotlib.pyplot as plt
#
# # 数据
# k = [2, 4, 6]
# data = {
#     "Modality1": {"MAE": [2.65, 2.67, 2.71], "RMSE": [4.78, 4.75, 4.82]},
#     "Modality2": {"MAE": [2.79, 2.79, 2.82], "RMSE": [5.08, 5.05, 5.10]},
#     "Modality3": {"MAE": [9.01, 8.80, 9.03], "RMSE": [19.33, 18.74, 19.33]},
#     "Modality4": {"MAE": [9.61, 9.41, 9.61], "RMSE": [18.65, 18.02, 18.52]},
# }
#
# # 每个 Modality 一幅图
# for modality, metrics in data.items():
#     fig, ax1 = plt.subplots(figsize=(6, 4))
#
#     # 左 y轴 (MAE)
#     ax1.plot(k, metrics["MAE"], color='gray', marker="o", label="MAE")
#     ax1.set_xlabel("k")
#     ax1.set_ylabel("MAE", color='gray')
#     ax1.tick_params(axis="y", labelcolor='gray')
#     ax1.set_xticks([2, 4, 6])   # ✅ 只保留 2,4,6
#
#     # 右 y轴 (RMSE)
#     ax2 = ax1.twinx()
#     ax2.plot(k, metrics["RMSE"], color="crimson", marker="s", label="RMSE")
#     ax2.set_ylabel("RMSE", color="crimson")
#     ax2.tick_params(axis="y", labelcolor="crimson")
#
#
#
#     # 凡例
#     lines_1, labels_1 = ax1.get_legend_handles_labels()
#     lines_2, labels_2 = ax2.get_legend_handles_labels()
#     plt.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")
#
#     plt.tight_layout()
#     plt.show()

import matplotlib.pyplot as plt

# 数据
L = [2, 4, 6, 8, 10]
data_L = {
    "Modality1": {"MAE": [2.78, 2.71, 2.70, 2.72, 2.67], "RMSE": [4.83, 4.83, 4.87, 4.86, 4.75]},
    "Modality2": {"MAE": [2.90, 2.82, 2.79, 2.81, 2.79], "RMSE": [5.14, 5.14, 5.11, 5.13, 5.05]},
    "Modality3": {"MAE": [9.14, 9.06, 9.13, 9.11, 8.80], "RMSE": [19.26, 19.26, 19.46, 19.43, 18.74]},
    "Modality4": {"MAE": [9.75, 9.67, 9.80, 9.62, 9.41], "RMSE": [18.57, 18.57, 19.10, 18.56, 18.34]},
}

# 每个 Modality 一幅图
for modality, metrics in data_L.items():
    fig, ax1 = plt.subplots(figsize=(6, 4))

    # 左 y轴 (MAE) - 蓝色
    ax1.plot(L, metrics["MAE"], color="royalblue", marker="o", label="MAE")
    ax1.set_xlabel("L")
    ax1.set_ylabel("MAE", color="royalblue")
    ax1.tick_params(axis="y", labelcolor="royalblue")
    ax1.set_xticks([2,4,6,8,10])   # ✅ 只保留 2,4,6,8,10

    # 右 y轴 (RMSE) - 橙色
    ax2 = ax1.twinx()
    ax2.plot(L, metrics["RMSE"], color="darkorange", marker="s", label="RMSE")
    ax2.set_ylabel("RMSE", color="darkorange")
    ax2.tick_params(axis="y", labelcolor="darkorange")



    # 合并图例
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    plt.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")

    plt.tight_layout()
    plt.show()

# import matplotlib.pyplot as plt
#
# # 数据
# p = [0.25, 0.5, 0.75]
# data_p = {
#     "Modality1": {"MAE": [2.67, 2.63, 2.66], "RMSE": [4.75, 4.73, 4.79]},
#     "Modality2": {"MAE": [2.79, 2.72, 2.76], "RMSE": [5.05, 5.00, 5.04]},
#     "Modality3": {"MAE": [8.80, 9.05, 8.93], "RMSE": [18.74, 19.36, 19.15]},
#     "Modality4": {"MAE": [9.41, 9.56, 9.58], "RMSE": [18.02, 18.47, 18.42]},
# }
#
# # 每个 Modality 一幅图
# for modality, metrics in data_p.items():
#     fig, ax1 = plt.subplots(figsize=(6, 4))
#
#     # 左 y轴 (MAE) - 绿色
#     ax1.plot(p, metrics["MAE"], color="seagreen", marker="o", label="MAE")
#     ax1.set_xlabel("p")
#     ax1.set_ylabel("MAE", color="seagreen")
#     ax1.tick_params(axis="y", labelcolor="seagreen")
#     ax1.set_xticks([0.25, 0.5, 0.75])  # ✅ 只保留 p=0.25,0.5,0.75
#
#     # 右 y轴 (RMSE) - 紫色
#     ax2 = ax1.twinx()
#     ax2.plot(p, metrics["RMSE"], color="purple", marker="s", label="RMSE")
#     ax2.set_ylabel("RMSE", color="purple")
#     ax2.tick_params(axis="y", labelcolor="purple")
#
#
#
#     # 合并图例
#     lines_1, labels_1 = ax1.get_legend_handles_labels()
#     lines_2, labels_2 = ax2.get_legend_handles_labels()
#     plt.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")
#
#     plt.tight_layout()
#     plt.show()