import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
def softmax_weighted_loss(output, target, numcls=None):
    target = target.float()
    B, _, H, W, Z = output.size()
    for i in range (numcls):
        outputi = output[:, i, :, :, :]
        targeti = target[:, i, :, :, :]
        weighted = 1.0 - (torch.sum(targeti, (1, 2, 3)) * 1.0 / torch.sum(target, (1, 2, 3, 4)))
        weighted = torch.reshape(weighted, (-1, 1, 1, 1)).repeat(1, H, W, Z)
        if i == 0:
            cross_loss = -1.0 * weighted * targeti * torch.log(torch.clamp(outputi, min=0.005, max=1)).float()
        else:
            cross_loss += -1.0 * weighted * targeti * torch.log(torch.clamp(outputi, min=0.005, max=1)).float()
    cross_loss = torch.mean(cross_loss)
    return cross_loss

def dice_loss(output, target, num_cls=None, eps=1e-7):
    target = target.float()
    for i in range(num_cls):
        num = torch.sum(output[:, i, :, :, :] * target[:, i, :, :, :])
        l = torch.sum(output[:, i, :, :, :])
        r = torch.sum(target[:, i, :, :, :])
        if i == 0:
            dice = 2.0 * num / (l+r+eps)
        else:
            dice += 2.0 * num / (l+r+eps)

    return 1.0 - 1.0 * dice / num_cls

def Binary_dice_loss(predictive, target, ep=1e-8):
    intersection = 2 * torch.sum(predictive * target) + ep
    union = torch.sum(predictive) + torch.sum(target) + ep
    loss = 1 - intersection / union
    return loss

def kl_loss(inputs, targets, ep=1e-8):
    kl_loss=nn.KLDivLoss(reduction='mean')
    consist_loss = kl_loss(torch.log(inputs+ep), targets)
    return consist_loss

def soft_ce_loss(inputs, target, ep=1e-8):
    logprobs = torch.log(inputs+ep)
    return  torch.mean(-(target[:,0,...]*logprobs[:,0,...]+target[:,1,...]*logprobs[:,1,...]))

def MSE_loss(input1, input2, num_cls=None):
    total_loss = 0.0
    for i in range(num_cls):
        mse_per_class = torch.mean((input1[:, i, ...] - input2[:, i, ...])**2)
        total_loss += mse_per_class
    return total_loss / num_cls
def mse_loss(input1, input2):
    return torch.mean((input1 - input2)**2)

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-10
        intersection = torch.sum(score * target)
        union = torch.sum(score * score) + torch.sum(target * target) + smooth
        loss = 1 - intersection / union
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


class HaarWaveletTransform3D(nn.Module):
    """
    3D Haar 小波变换 (离散小波变换 DWT)
    不需要第三方库，基于 PyTorch 切片实现，支持反向传播。
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        B, C, D, H, W = x.shape

        # 填充以确保尺寸是偶数
        pad_d = D % 2
        pad_h = H % 2
        pad_w = W % 2
        if pad_d + pad_h + pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d), mode='replicate')

        # 提取 8 个子带
        x0 = x[:, :, 0::2, 0::2, 0::2]
        x1 = x[:, :, 0::2, 0::2, 1::2]
        x2 = x[:, :, 0::2, 1::2, 0::2]
        x3 = x[:, :, 0::2, 1::2, 1::2]
        x4 = x[:, :, 1::2, 0::2, 0::2]
        x5 = x[:, :, 1::2, 0::2, 1::2]
        x6 = x[:, :, 1::2, 1::2, 0::2]
        x7 = x[:, :, 1::2, 1::2, 1::2]

        # Haar 变换
        LLL = (x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7) / 8.0  # 低频

        # 高频部分
        LLH = (x0 + x1 + x2 + x3 - x4 - x5 - x6 - x7) / 8.0
        LHL = (x0 + x1 - x2 - x3 + x4 + x5 - x6 - x7) / 8.0
        LHH = (x0 + x1 - x2 - x3 - x4 - x5 + x6 + x7) / 8.0
        HLL = (x0 - x1 + x2 - x3 + x4 - x5 + x6 - x7) / 8.0
        HLH = (x0 - x1 + x2 - x3 - x4 + x5 - x6 + x7) / 8.0
        HHL = (x0 - x1 - x2 + x3 + x4 - x5 - x6 + x7) / 8.0
        HHH = (x0 - x1 - x2 + x3 - x4 + x5 + x6 - x7) / 8.0

        # 将所有高频分量拼接
        HF = torch.cat([LLH, LHL, LHH, HLL, HLH, HHL, HHH], dim=1)

        return LLL, HF


class FreqConsistencyLoss(nn.Module):
    """
    置信度感知的频域一致性损失 (支持排除背景类)
    """

    def __init__(self, epsilon=1e-6, ignore_index=0):
        super().__init__()
        self.dwt = HaarWaveletTransform3D()
        self.epsilon = epsilon
        # 注意：这里 reduction='none' 是必须的，我们需要手动处理类别维度
        self.criterion = nn.L1Loss(reduction='none')
        self.ignore_index = ignore_index

    def forward(self, logits_A, logits_B):
        # 1. 获取概率分布 [B, C, D, H, W]
        prob_A = F.softmax(logits_A, dim=1)
        prob_B = F.softmax(logits_B, dim=1)

        num_classes = prob_A.shape[1]

        # 2. 计算置信度 (Confidence)
        conf_A, _ = torch.max(prob_A, dim=1, keepdim=True)
        conf_B, _ = torch.max(prob_B, dim=1, keepdim=True)

        # 3. 计算相对权重
        w_A = conf_A / (conf_A + conf_B + self.epsilon)
        w_B = 1.0 - w_A

        # 下采样权重以匹配 DWT 输出尺寸 [B, 1, D/2, H/2, W/2]
        w_A_down = F.avg_pool3d(w_A, kernel_size=2, stride=2)
        w_B_down = F.avg_pool3d(w_B, kernel_size=2, stride=2)

        # 4. DWT 分解
        # LL: [B, C, D/2, ...]
        # HF: [B, 7*C, D/2, ...]
        LL_A, HF_A = self.dwt(prob_A)
        LL_B, HF_B = self.dwt(prob_B)

        # ================== 低频损失计算 (LL) ==================
        # 形状: [B, C, D/2, H/2, W/2]
        diff_LL = torch.abs(LL_A.detach() - LL_B) * w_A_down + \
                  torch.abs(LL_B.detach() - LL_A) * w_B_down

        # 空间维度求平均 -> [B, C]
        loss_LL_per_class = diff_LL.mean(dim=(2, 3, 4))

        # ================== 高频损失计算 (HF) ==================
        # 原始 HF 是 [B, 7*C, ...], 我们将其 reshape 为 [B, 7, C, ...] 以便分离 C
        B, _, D_half, H_half, W_half = HF_A.shape
        HF_A_reshaped = HF_A.view(B, 7, num_classes, D_half, H_half, W_half)
        HF_B_reshaped = HF_B.view(B, 7, num_classes, D_half, H_half, W_half)

        # 同样需要扩展权重维度到 [B, 1, 1, D/2, ...]
        w_A_exp = w_A_down.unsqueeze(1)  # [B, 1, 1, D/2, ...]
        w_B_exp = w_B_down.unsqueeze(1)

        diff_HF = torch.abs(HF_A_reshaped.detach() - HF_B_reshaped) * w_A_exp + \
                  torch.abs(HF_B_reshaped.detach() - HF_A_reshaped) * w_B_exp

        # 空间维度 + 7个频带维度求平均 -> [B, C]
        loss_HF_per_class = diff_HF.mean(dim=(1, 3, 4, 5))

        # ================== 排除背景类并求平均 ==================
        if self.ignore_index is not None:
            # 创建一个 mask，选出不是 ignore_index 的类别 (即 1, 2, 3)
            # indices: [0, 1, 2, 3]
            class_indices = torch.arange(num_classes).to(prob_A.device)
            valid_mask = (class_indices != self.ignore_index).float()  # [0, 1, 1, 1]

            # 乘以 mask，背景类 loss 变为 0
            loss_LL_per_class = loss_LL_per_class * valid_mask
            loss_HF_per_class = loss_HF_per_class * valid_mask

            # 求平均时，分母是有效类别的数量 (比如 3)
            num_valid = valid_mask.sum()
            loss_low = loss_LL_per_class.sum() / (num_valid + 1e-6)
            loss_high = loss_HF_per_class.sum() / (num_valid + 1e-6)
        else:
            # 如果不排除，直接求平均
            loss_low = loss_LL_per_class.mean()
            loss_high = loss_HF_per_class.mean()

        return loss_low, loss_high


class PrototypeMaskGenerator(nn.Module):
    """
    创新1: 基于原型的语义可靠性掩码
    通过维护每个类别的"平均概率分布原型"，过滤掉那些预测分布异常的像素。
    """

    def __init__(self, feature_dim, num_classes, momentum=0.99):
        super().__init__()
        self.num_classes = num_classes
        self.momentum = momentum
        # 初始化原型 [Class, Dim]
        self.register_buffer("prototypes", torch.zeros(num_classes, feature_dim))
        self.register_buffer("initialized", torch.zeros(1))

    def update_prototypes(self, features, pred_softmax):
        # features: 这里我们使用 logits 作为特征 [B, C, D, H, W]
        # pred_softmax: [B, C, D, H, W]

        pseudo_label = torch.argmax(pred_softmax, dim=1)

        with torch.no_grad():
            for c in range(self.num_classes):
                # 找到属于该类的像素 (为了省显存，每张图最多采样 1000 个点)
                mask_c = (pseudo_label == c)
                if mask_c.sum() < 10: continue

                # 提取特征并计算平均
                # [B, C, D, H, W] -> permute -> [B, D, H, W, C] -> mask -> [N, C]
                feat_c = features.permute(0, 2, 3, 4, 1)[mask_c]

                # 如果点太多，随机采样以节省计算
                if feat_c.shape[0] > 1000:
                    indices = torch.randperm(feat_c.shape[0])[:1000]
                    feat_c = feat_c[indices]

                center_c = feat_c.mean(dim=0)
                center_c = F.normalize(center_c, dim=0)

                if self.initialized.item() == 0:
                    self.prototypes[c] = center_c
                else:
                    self.prototypes[c] = self.momentum * self.prototypes[c] + (1 - self.momentum) * center_c

            self.initialized.fill_(1)
            self.prototypes = F.normalize(self.prototypes, dim=1)

    def forward(self, features, pred_softmax):
        """
        计算语义可靠性掩码
        features: Logits [B, C, D, H, W]
        """
        with torch.no_grad():
            B, C, D, H, W = features.shape
            pseudo_label = torch.argmax(pred_softmax, dim=1)

            # 归一化特征
            features_flat = features.permute(0, 2, 3, 4, 1).reshape(-1, C)
            features_flat = F.normalize(features_flat, dim=1)

            # 获取对应类别的原型
            target_protos = self.prototypes[pseudo_label.view(-1)]  # [N, C]

            # 计算余弦相似度
            similarity = (features_flat * target_protos).sum(dim=1).view(B, 1, D, H, W)

            # 映射到 0~1 (相似度越高，mask 越接近 1)
            semantic_mask = torch.clamp(similarity, 0, 1)

            return semantic_mask


def compute_boundary_aware_mask(logits, uncertainty_map):
    """
    创新2: 边界感知掩码
    结合 Entropy (熵) 和 Uncertainty (方差)
    """
    with torch.no_grad():
        prob = F.softmax(logits, dim=1)

        # 1. 计算信息熵 (Entropy): 衡量预测的混乱程度，边界处熵最大
        # -sum(p * log(p))
        entropy = -torch.sum(prob * torch.log(prob + 1e-6), dim=1, keepdim=True)
        # 归一化熵到 0~1
        entropy = entropy / np.log(prob.shape[1])

        # 2. 基础方差掩码 (信任方差小的区域)
        # 这里的系数 10.0 可以根据实际调整
        variance_mask = torch.exp(-10.0 * uncertainty_map)

        # 3. 边界松弛策略
        # 在熵高(边界)的地方，即便方差大，也给予一定的权重(0.5)，防止边界被完全忽略
        boundary_factor = torch.sigmoid(entropy * 5.0)

        # 组合：基础掩码 + 边界补偿
        final_mask = variance_mask + 0.3 * boundary_factor
        final_mask = torch.clamp(final_mask, 0, 1)

        return final_mask