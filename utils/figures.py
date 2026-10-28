# 显示输入图片并保存
import numpy as np
import matplotlib.pyplot as plt
import os
from dataloaders.transforms import RandCrop3D

# ==========================
# 1. 设置保存路径
# ==========================
save_dir = "/data1/zht/SeSuInMo_Net_main/results/input_figures"  # 图片保存的文件夹
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# ==========================
# 2. 载入与预处理数据
# ==========================
# 载入数据
data_path = "/data1/zht/SeSuInMo_Net_main/data/BraTS2018_tar/BraTS_Test/vol/HGG_Brats18_2013_10_1_vol.npy"
data = np.load(data_path)

# 根据你的注释 (C, H, W, D) 进行转置
# 原始数据通常是 (H, W, D, C)，转置后为 (C, D, W, H) 或者 (C, H, W, D)，这里沿用你的逻辑
data = np.transpose(data, (3, 2, 1, 0))

# 定义模态名称
modalities = ["FLAIR", "T1ce", "T1", "T2"]

# 获取前4个模态 (C, H, W, D)
images = data[:4]

# 随机裁剪
# crop_size = (128, 128, 128)
# rand_crop = RandCrop3D(crop_size)
# cropped_images = rand_crop(images)  # Shape: (4, 128, 128, 128)

# ==========================
# 3. 提取切片并保存
# ==========================
# 选择切片位置：这里取深度方向(D)的中间位置，得到一张 (H, W) 的图
# 如果你想切其他方向，可以修改索引位置，例如 cropped_images[i, slice_idx, :, :]
slice_idx = images.shape[-1] // 2
# slice_idx = 56

plt.figure(figsize=(16, 4))  # 设置画布大小

for i, mod_name in enumerate(modalities):
    # 提取第 i 个模态的 2D 切片
    # 假设 shape 是 (C, H, W, D)，我们取最后一个维度 D 的切片
    image_slice = images[i, slice_idx, :, :, ]

    # --- 显示 ---
    plt.subplot(1, 4, i + 1)
    plt.imshow(image_slice, cmap='gray')
    plt.title(mod_name)
    plt.axis('off')

    # --- 保存 ---
    # 保存为单独的文件，文件名为 模态名_slice.png
    save_path = os.path.join(save_dir, f"{mod_name}_slice.tiff")

    # 使用 plt.imsave 可以直接保存数组为图片，cmap='gray' 保证是灰度图
    plt.imsave(save_path, image_slice, cmap='gray')
    print(f"已保存: {save_path}")

plt.tight_layout()
plt.show()
#
# 保存label
# import numpy as np
# import matplotlib.pyplot as plt
# import os
#
# # ==========================
# # 1. 设置文件路径
# # ==========================
# seg_path = "/data1/zht/SeSuInMo_Net_main/data/BraTS2018_tar/BraTS_Test/seg/HGG_Brats18_2013_10_1_seg.npy"
# save_path = "/data1/zht/SeSuInMo_Net_main/results/input_figures/GroundTruth_Label_Color_Pretty.tiff"
#
# # ==========================
# # 2. 载入与预处理
# # ==========================
# print(f"正在载入 Label: {seg_path}")
# seg_data = np.load(seg_path)
#
# # 维度调整
# seg_data = np.transpose(seg_data, (2, 1, 0))
# print(f"Label Shape: {seg_data.shape}")
#
# # ==========================
# # 3. 选择切片并智能上色
# # ==========================
# # 自动寻找包含增强肿瘤(Label 4)最多的切片，如果没有4则找中间
# target_pixels = np.sum(seg_data == 4, axis=(1, 2))
# if np.max(target_pixels) > 0:
#     slice_idx = np.argmax(target_pixels)
#     print(f"自动选择增强区域最大的切片: 第 {slice_idx} 层")
# else:
#     slice_idx = seg_data.shape[0] // 2
#     print(f"未检测到增强区域，选择中间切片: 第 {slice_idx} 层")
#
# label_slice = seg_data[slice_idx, :, :]
#
# # 创建 RGB 画布
# h, w = label_slice.shape
# rgb_label = np.zeros((h, w, 3), dtype=np.float32)
#
# # =======================================================
# # 【核心修改】 配色方案配置
# # =======================================================
#
# # --- 方案 A: 优雅论文风 (推荐) ---
# # 更加柔和的红、青绿、金黄，适合发表文章
# colors = {
#     1: [0.94, 0.35, 0.35],  # 坏死 (NCR): 柔和珊瑚红
#     2: [0.30, 0.85, 0.70],  # 水肿 (Edema): 绿松石色/青绿
#     4: [1.00, 0.85, 0.30],  # 增强 (ET): 金黄色
#     3: [1.00, 0.85, 0.30]   # 兼容 Label 3
# }
#
# # --- 方案 B: 赛博朋克风 (备选，取消注释使用) ---
# # colors = {
# #     1: [0.2, 0.2, 0.9],   # 坏死: 深蓝
# #     2: [0.0, 0.9, 0.9],   # 水肿: 青色 (Cyan)
# #     4: [1.0, 0.2, 0.6],   # 增强: 玫红 (Magenta)
# #     3: [1.0, 0.2, 0.6]
# # }
#
# # --- 方案 C: ITK-SNAP 经典风 (备选，取消注释使用) ---
# # colors = {
# #     1: [1.0, 0.0, 0.0],   # 红
# #     2: [0.0, 1.0, 0.0],   # 绿
# #     4: [0.0, 0.0, 1.0],   # 蓝 (ITK-SNAP常用蓝色表示增强)
# #     3: [0.0, 0.0, 1.0]
# # }
#
# # =======================================================
#
# # 应用颜色
# for label_val, color_rgb in colors.items():
#     # 找到对应标签的像素位置，并赋值颜色
#     rgb_label[label_slice == label_val] = color_rgb
#
# # ==========================
# # 4. 保存
# # ==========================
# # 确保父目录存在
# os.makedirs(os.path.dirname(save_path), exist_ok=True)
#
# plt.imsave(save_path, rgb_label)
# print(f"已保存优化配色的标签至: {save_path}")





