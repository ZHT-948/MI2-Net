import os
import glob
import numpy as np
import medpy.io as medio  # pip install medpy
import math

join = os.path.join

# ================= 配置路径 =================
# 请根据实际情况修改
src_path = '/data0/zht/data/MyoPS2020_ori'
tar_path = '/data0/zht/data/MyoPS2020_tar'

# 目标 Z 轴对齐倍数 (Mamba/U-Net 4次下采样需要能被16整除)
Z_DIVISOR = 16

# 获取所有 Case 的文件夹名称
if os.path.exists(src_path):
    case_list = [d for d in os.listdir(src_path) if os.path.isdir(join(src_path, d))]
    case_list.sort()
else:
    case_list = []
    print(f"Error: Source path {src_path} not found.")


# ================= 辅助函数 =================

def sup_288(xmin, xmax):
    """
    确保裁剪区域在 XY 平面上至少有 288 的长度
    """
    if xmax - xmin < 288:
        ecart = int((288 - (xmax - xmin)) / 2)
        xmax = xmax + ecart + 1
        xmin = xmin - ecart
    if xmin < 0:
        xmax -= xmin
        xmin = 0
    return xmin, xmax


def pad_z_to_multiple(vol, seg, divisor=16):
    """
    将 Z 轴 (Depth) Padding 到 divisor 的整数倍 (例如 16)
    vol shape: (C, H, W, D)
    seg shape: (H, W, D)
    """
    current_d = vol.shape[3]  # Z轴在第3维

    # 计算目标层数：如果是 5，目标是 16；如果是 17，目标是 32
    if current_d < divisor:
        target_d = divisor
    else:
        target_d = math.ceil(current_d / divisor) * divisor

    if current_d == target_d:
        return vol, seg

    pad_total = target_d - current_d
    pad_front = pad_total // 2
    pad_back = pad_total - pad_front

    # 打印提示，方便调试
    # print(f"    Padding Z from {current_d} to {target_d} (Front: {pad_front}, Back: {pad_back})")

    # np.pad 格式: ((before_axis0, after_axis0), ...)
    # vol: (C, H, W, D) -> 只 Pad 最后一个维度
    pad_width_vol = ((0, 0), (0, 0), (0, 0), (pad_front, pad_back))

    # seg: (H, W, D) -> 只 Pad 最后一个维度
    pad_width_seg = ((0, 0), (0, 0), (pad_front, pad_back))

    # 使用 constant=0 进行填充 (背景)
    vol_padded = np.pad(vol, pad_width_vol, mode='constant', constant_values=0)
    seg_padded = np.pad(seg, pad_width_seg, mode='constant', constant_values=0)

    return vol_padded, seg_padded


def crop(vol):
    """
    计算裁剪坐标。
    XY 平面保证 288，Z 轴仅紧致裁剪 (tight crop)，后续通过 Pad 补齐。
    """
    if len(vol.shape) == 4:
        vol_mask = np.sum(vol, axis=0) > 0
    else:
        vol_mask = vol > 0

    x_nonzeros, y_nonzeros, z_nonzeros = np.where(vol_mask)

    if len(x_nonzeros) == 0:
        return 0, vol.shape[1], 0, vol.shape[2], 0, vol.shape[3]

    x_min, x_max = np.amin(x_nonzeros), np.amax(x_nonzeros)
    y_min, y_max = np.amin(y_nonzeros), np.amax(y_nonzeros)
    z_min, z_max = np.amin(z_nonzeros), np.amax(z_nonzeros)

    # XY 强制扩展到 288
    x_min, x_max = sup_288(x_min, x_max)
    y_min, y_max = sup_288(y_min, y_max)

    # Z 轴保持紧致裁剪 (不扩展)，留给 pad_z_to_multiple 处理
    # z_min, z_max 保持原始 ROI 即可，稍微留 1 像素缓冲可选，这里保持 tight
    return x_min, x_max, y_min, y_max, z_min, z_max


def normalize(vol):
    """
    Z-score 归一化。
    !!!重要修改!!!:
    之前的代码如果直接 (x - mean)/std，会导致原本为 0 的背景变成负数。
    这会导致后续 Pad 0 时出现 artifacts。
    这里修改为：只改变前景像素值，背景保持严格为 0。
    """
    mask = vol.sum(0) > 0
    for k in range(vol.shape[0]):
        x = vol[k, ...]
        y = x[mask]  # 取出前景

        if y.size > 0 and y.std() > 0:
            mean = y.mean()
            std = y.std()
            # 仅对前景做归一化
            x[mask] = (x[mask] - mean) / std
            # 强制背景为 0 (防止浮点误差)
            x[~mask] = 0

        vol[k, ...] = x

    return vol


# ================= 主处理流程 =================

if not os.path.exists(os.path.join(tar_path, 'vol')):
    os.makedirs(os.path.join(tar_path, 'vol'))
if not os.path.exists(os.path.join(tar_path, 'seg')):
    os.makedirs(os.path.join(tar_path, 'seg'))

print(f"Total cases found: {len(case_list)}")

for case_name in case_list:
    print(f"Processing: {case_name}")
    case_path = join(src_path, case_name)

    try:
        c0_path = glob.glob(join(case_path, '*C0*.nii*'))[0]
        lge_path = glob.glob(join(case_path, '*LGE*.nii*'))
        if not lge_path: lge_path = glob.glob(join(case_path, '*DE*.nii*'))
        lge_path = lge_path[0]
        t2_path = glob.glob(join(case_path, '*T2*.nii*'))[0]
        seg_path = glob.glob(join(case_path, '*gd*.nii*'))[0]
    except IndexError:
        print(f"Skipping {case_name}: Files missing.")
        continue

    # 1. 读取
    c0, _ = medio.load(c0_path)
    lge, _ = medio.load(lge_path)
    t2, _ = medio.load(t2_path)
    seg, _ = medio.load(seg_path)

    # 2. 堆叠 (Channels, H, W, D)
    vol = np.stack((c0, lge, t2), axis=0).astype(np.float32)

    # 3. 裁剪 (XY fixed, Z tight)
    x_min, x_max, y_min, y_max, z_min, z_max = crop(vol)

    # 边界保护
    x_min = max(0, x_min);
    y_min = max(0, y_min);
    z_min = max(0, z_min)
    x_max = min(vol.shape[1], x_max);
    y_max = min(vol.shape[2], y_max);
    z_max = min(vol.shape[3], z_max + 1)

    vol_cropped = vol[:, x_min:x_max, y_min:y_max, z_min:z_max]
    seg_cropped = seg[x_min:x_max, y_min:y_max, z_min:z_max]

    # 4. 归一化 (背景保持 0)
    vol_norm = normalize(vol_cropped)

    # 5. 标签映射 (先映射，因为 Padding 补的是 0，正好对应背景)
    seg_out = np.zeros_like(seg_cropped, dtype=np.uint8)
    seg_out[seg_cropped == 200] = 1
    seg_out[seg_cropped == 500] = 2
    seg_out[seg_cropped == 600] = 3
    seg_out[seg_cropped == 1220] = 4
    seg_out[seg_cropped == 2221] = 5

    # 6. !!! Padding Z轴 !!!
    # 这里将 Z 轴填充到 16 的倍数
    vol_padded, seg_padded = pad_z_to_multiple(vol_norm, seg_out, divisor=Z_DIVISOR)

    # 7. 转换维度 (C, H, W, D) -> (H, W, D, C)
    # PyTorch DataLoader 通常需要 .npy 为 (H, W, D, C)
    # 注意：你的代码之前是 transpose(1, 2, 3, 0)，这代表 H, W, D, C
    vol_final = vol_padded.transpose(1, 2, 3, 0)
    seg_final = seg_padded  # (H, W, D)

    # 8. 保存
    print(f"  Ori Z: {z_max - z_min} -> Padded Z: {vol_final.shape[2]} | Final Shape: {vol_final.shape}")
    np.save(os.path.join(tar_path, 'vol', f'{case_name}_vol.npy'), vol_final)
    np.save(os.path.join(tar_path, 'seg', f'{case_name}_seg.npy'), seg_final)

print("Preprocessing Done!")