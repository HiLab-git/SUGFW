import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

# 数据目录
data_dir = "/media/ubuntu/maxiaochuan/MA-SAM/data/Promise12/2D_slices"
files = os.listdir(data_dir)

# 定义一个函数来归一化数据
def normalize_data(data):
    """
    Normalize the data to the range [0, 1].
    
    Args:
        data (np.ndarray): Input 2D array.
    
    Returns:
        np.ndarray: Normalized 2D array.
    """
    return (data - np.min(data)) / (np.max(data) - np.min(data))

# 定义一个函数来保存合并的热力图
def save_combined_heatmap(data_list, titles, filename):
    """
    Save a combined heatmap of multiple data arrays.

    Args:
        data_list (list of np.ndarray): List of 2D arrays to plot.
        titles (list of str): Titles for each subplot.
        filename (str): Output file name for the combined heatmap.
    """
    num_plots = len(data_list)
    plt.figure(figsize=(12, 12))  # 调整图片大小
    for i, data in enumerate(data_list):
        plt.subplot(2, 2, i + 1)  # 创建2x2子图
        plt.imshow(data, cmap="hot", interpolation="nearest")
        plt.colorbar(label="Uncertainty")
        plt.title(titles[i])
        plt.axis("off")  # 隐藏坐标轴
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight", pad_inches=0)
    plt.close()

# 输出目录
output_dir = "uncertainty_contrast"
os.makedirs(output_dir, exist_ok=True)

# 遍历文件并处理符合条件的样本
for file in files:
    if file.startswith("Case"):
        case_number = int(file[4:6])  # 提取 Case 后的数字
        if case_number < 50:
            base_name = file.split("_")[0] + "_" + file.split("_")[1]  # 提取文件的基础名称
            u1 = os.path.join(data_dir, f"{base_name}_uncertainty_5.pkl")
            u2 = os.path.join(data_dir, f"{base_name}_uncertainty_10.pkl")
            u3 = os.path.join(data_dir, f"{base_name}_uncertainty_15.pkl")
            u4 = os.path.join(data_dir, f"{base_name}_uncertainty_20.pkl")

            # 加载数据
            with open(u1, "rb") as f:
                u1_data = pickle.load(f)
            with open(u2, "rb") as f:
                u2_data = pickle.load(f)
            with open(u3, "rb") as f:
                u3_data = pickle.load(f)
            with open(u4, "rb") as f:
                u4_data = pickle.load(f)

            # 归一化数据
            u1_data_normalized = normalize_data(u1_data)
            u2_data_normalized = normalize_data(u2_data)
            u3_data_normalized = normalize_data(u3_data)
            u4_data_normalized = normalize_data(u4_data)

            # 保存合并的热力图
            save_combined_heatmap(
                [u1_data_normalized, u2_data_normalized, u3_data_normalized, u4_data_normalized],
                ["Uncertainty 5", "Uncertainty 10", "Uncertainty 15", "Uncertainty 20"],
                f"{output_dir}/{base_name}_combined_heatmap.png",
            )