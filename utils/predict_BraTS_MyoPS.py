import csv
import os
import time
import logging
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from medpy.metric import hd95
import torch.nn.functional as F

cudnn.benchmark = True

path = os.path.dirname(__file__)

patch_size1 = 128
patch_size_xy = 288
patch_size_z = 16
patch_size_RFNet = 80

def compute_BraTS_HD95(ref, pred):
    """
    ref and gt are binary integer numpy.ndarray s
    spacing is assumed to be (1, 1, 1)
    :param ref:
    :param pred:
    :return:
    """
    num_ref = np.sum(ref)
    num_pred = np.sum(pred)
    if num_ref == 0:
        if num_pred == 0:
            return 0
        else:
            return 1.0
            # follow ACN and SMU-Net
            # return 373.12866
            # follow nnUNet
    elif num_pred == 0 and num_ref != 0:
        return 1.0
        # follow ACN and SMU-Net
        # return 373.12866
        # follow in nnUNet
    else:
        return hd95(pred, ref, (1, 1, 1))

def cal_hd95(output, target):
    # whole tumor
    mask_gt = (target != 0).astype(int)
    mask_pred = (output != 0).astype(int)
    hd95_whole = compute_BraTS_HD95(mask_gt, mask_pred)
    del mask_gt, mask_pred

    # tumor core
    mask_gt = ((target == 1) | (target == 3)).astype(int)
    mask_pred = ((output == 1) | (output == 3)).astype(int)
    hd95_core = compute_BraTS_HD95(mask_gt, mask_pred)
    del mask_gt, mask_pred

    # enhancing
    mask_gt = (target == 3).astype(int)
    mask_pred = (output == 3).astype(int)
    hd95_enh = compute_BraTS_HD95(mask_gt, mask_pred)
    del mask_gt, mask_pred

    mask_gt = (target == 3).astype(int)
    if np.sum((output == 3).astype(int)) < 500:
        mask_pred = (output == 3).astype(int) * 0
    else:
        mask_pred = (output == 3).astype(int)
    hd95_enhpro = compute_BraTS_HD95(mask_gt, mask_pred)
    del mask_gt, mask_pred

    return (hd95_whole, hd95_core, hd95_enh)


def compute_MyoPS_HD95(ref, pred):
    """
    ref and pred are binary integer numpy.ndarray s
    spacing is assumed to be (1, 1, 1) or handled by the external hd95 function
    """
    num_ref = np.sum(ref)
    num_pred = np.sum(pred)

    if num_ref == 0:
        if num_pred == 0:
            return 0
        else:
            return 1.0
    elif num_pred == 0 and num_ref != 0:
        return 1.0
    else:
        return hd95(pred, ref, voxelspacing=(1, 1, 1))


def cal_hd95_MyoPS(output, target):
    """
    计算 Myo, MI, MI+ME, LV, RV 的 HD95
    Input: numpy array (int)
    """

    # --- 1. Myo (1 + 4 + 5) ---
    mask_gt = ((target == 1) | (target == 4) | (target == 5)).astype(int)
    mask_pred = ((output == 1) | (output == 4) | (output == 5)).astype(int)
    hd95_myo = compute_MyoPS_HD95(mask_gt, mask_pred)
    del mask_gt, mask_pred

    # --- 2. MI (5) ---
    mask_gt = (target == 5).astype(int)
    mask_pred = (output == 5).astype(int)
    hd95_mi = compute_MyoPS_HD95(mask_gt, mask_pred)
    del mask_gt, mask_pred

    # --- 3. MI + ME (4 + 5) ---
    mask_gt = ((target == 4) | (target == 5)).astype(int)
    mask_pred = ((output == 4) | (output == 5)).astype(int)
    hd95_mime = compute_MyoPS_HD95(mask_gt, mask_pred)
    del mask_gt, mask_pred

    # --- 4. LV (2) ---
    mask_gt = (target == 2).astype(int)
    mask_pred = (output == 2).astype(int)
    hd95_lv = compute_MyoPS_HD95(mask_gt, mask_pred)
    del mask_gt, mask_pred

    # --- 5. RV (3) ---
    mask_gt = (target == 3).astype(int)
    mask_pred = (output == 3).astype(int)
    hd95_rv = compute_MyoPS_HD95(mask_gt, mask_pred)
    del mask_gt, mask_pred

    return (hd95_myo, hd95_mi, hd95_mime, hd95_lv, hd95_rv)

def softmax_output_dice_class4(output, target):
    eps = 1e-8
    #######label1########
    o1 = (output == 1).float()
    t1 = (target == 1).float()
    intersect1 = torch.sum(2 * (o1 * t1), dim=(1, 2, 3)) + eps
    denominator1 = torch.sum(o1, dim=(1, 2, 3)) + torch.sum(t1, dim=(1, 2, 3)) + eps
    ncr_net_dice = intersect1 / denominator1

    o2 = (output == 2).float()
    t2 = (target == 2).float()
    intersect2 = torch.sum(2 * (o2 * t2), dim=(1, 2, 3)) + eps
    denominator2 = torch.sum(o2, dim=(1, 2, 3)) + torch.sum(t2, dim=(1, 2, 3)) + eps
    edema_dice = intersect2 / denominator2

    o3 = (output == 3).float()
    t3 = (target == 3).float()
    intersect3 = torch.sum(2 * (o3 * t3), dim=(1, 2, 3)) + eps
    denominator3 = torch.sum(o3, dim=(1, 2, 3)) + torch.sum(t3, dim=(1, 2, 3)) + eps
    enhancing_dice = intersect3 / denominator3

    ####post processing:
    if torch.sum(o3) < 500:
       o4 = o3 * 0.0
    else:
       o4 = o3
    t4 = t3
    intersect4 = torch.sum(2 * (o4 * t4), dim=(1, 2, 3)) + eps
    denominator4 = torch.sum(o4, dim=(1, 2, 3)) + torch.sum(t4, dim=(1, 2, 3)) + eps
    enhancing_dice_postpro = intersect4 / denominator4

    o_whole = o1 + o2 + o3
    t_whole = t1 + t2 + t3
    intersect_whole = torch.sum(2 * (o_whole * t_whole), dim=(1, 2, 3)) + eps
    denominator_whole = torch.sum(o_whole, dim=(1, 2, 3)) + torch.sum(t_whole, dim=(1, 2, 3)) + eps
    dice_whole = intersect_whole / denominator_whole

    o_core = o1 + o3
    t_core = t1 + t3
    intersect_core = torch.sum(2 * (o_core * t_core), dim=(1, 2, 3)) + eps
    denominator_core = torch.sum(o_core, dim=(1, 2, 3)) + torch.sum(t_core, dim=(1, 2, 3)) + eps
    dice_core = intersect_core / denominator_core


    dice_separate = torch.cat((torch.unsqueeze(ncr_net_dice, 1), torch.unsqueeze(edema_dice, 1), torch.unsqueeze(enhancing_dice, 1)), dim=1)
    # dice_evaluate = torch.cat((torch.unsqueeze(dice_whole, 1), torch.unsqueeze(dice_core, 1), torch.unsqueeze(enhancing_dice, 1)), dim=1)
    dice_evaluate = torch.cat(
        (torch.unsqueeze(dice_whole, 1), torch.unsqueeze(dice_core, 1), torch.unsqueeze(enhancing_dice, 1)), dim=1)

    return dice_separate.cpu().numpy(), dice_evaluate.cpu().numpy()

def softmax_output_dice_MyoPS(output, target):
    """
    MyoPS 标签映射:
    0: BG
    1: LV Myocardium (Normal)
    2: LV Blood Pool
    3: RV Blood Pool
    4: LV Myocardial Edema
    5: LV Myocardial Scars
    """
    eps = 1e-8

    # 确保 target 和 output 类型一致 (通常是 float 用于计算)
    # output 应该是 argmax 之后的标签图，或者是 one-hot 后的。
    # 根据你之前的代码上下文，这里的 output 是已经 argmax 后的 [B, H, W, D] 整数张量

    # --- 1. Myo Dice (Class 1 + 4 + 5) ---
    o_myo = ((output == 1) | (output == 4) | (output == 5)).float()
    t_myo = ((target == 1) | (target == 4) | (target == 5)).float()

    intersect_myo = torch.sum(2 * (o_myo * t_myo), dim=(1, 2, 3)) + eps
    denominator_myo = torch.sum(o_myo, dim=(1, 2, 3)) + torch.sum(t_myo, dim=(1, 2, 3)) + eps
    dice_myo = intersect_myo / denominator_myo

    # --- 2. MI Dice (Class 5) ---
    o_mi = (output == 5).float()
    t_mi = (target == 5).float()

    intersect_mi = torch.sum(2 * (o_mi * t_mi), dim=(1, 2, 3)) + eps
    denominator_mi = torch.sum(o_mi, dim=(1, 2, 3)) + torch.sum(t_mi, dim=(1, 2, 3)) + eps
    dice_mi = intersect_mi / denominator_mi

    # --- 3. MI + ME Dice (Class 4 + 5) ---
    o_mime = ((output == 4) | (output == 5)).float()
    t_mime = ((target == 4) | (target == 5)).float()

    intersect_mime = torch.sum(2 * (o_mime * t_mime), dim=(1, 2, 3)) + eps
    denominator_mime = torch.sum(o_mime, dim=(1, 2, 3)) + torch.sum(t_mime, dim=(1, 2, 3)) + eps
    dice_mime = intersect_mime / denominator_mime

    # --- 4. LV Dice (Class 2) ---
    o_lv = (output == 2).float()
    t_lv = (target == 2).float()

    intersect_lv = torch.sum(2 * (o_lv * t_lv), dim=(1, 2, 3)) + eps
    denominator_lv = torch.sum(o_lv, dim=(1, 2, 3)) + torch.sum(t_lv, dim=(1, 2, 3)) + eps
    dice_lv = intersect_lv / denominator_lv

    # --- 5. RV Dice (Class 3) ---
    o_rv = (output == 3).float()
    t_rv = (target == 3).float()

    intersect_rv = torch.sum(2 * (o_rv * t_rv), dim=(1, 2, 3)) + eps
    denominator_rv = torch.sum(o_rv, dim=(1, 2, 3)) + torch.sum(t_rv, dim=(1, 2, 3)) + eps
    dice_rv = intersect_rv / denominator_rv

    # 组合输出 [B, 5]
    dice_evaluate = torch.cat([
        torch.unsqueeze(dice_myo, 1),  # Index 0: Myo
        torch.unsqueeze(dice_mi, 1),  # Index 1: MI
        torch.unsqueeze(dice_mime, 1),  # Index 2: MI+ME
        torch.unsqueeze(dice_lv, 1),  # Index 3: LV
        torch.unsqueeze(dice_rv, 1)  # Index 4: RV
    ], dim=1)

    # 保持 separate 和 evaluate 一致，或者根据需求修改
    dice_separate = dice_evaluate

    return dice_separate.cpu().numpy(), dice_evaluate.cpu().numpy()

def softmax_output_dice_class5(output, target):
    eps = 1e-8
    #######label1########
    o1 = (output == 1).float()
    t1 = (target == 1).float()
    intersect1 = torch.sum(2 * (o1 * t1), dim=(1, 2, 3)) + eps
    denominator1 = torch.sum(o1, dim=(1, 2, 3)) + torch.sum(t1, dim=(1, 2, 3)) + eps
    necrosis_dice = intersect1 / denominator1

    o2 = (output == 2).float()
    t2 = (target == 2).float()
    intersect2 = torch.sum(2 * (o2 * t2), dim=(1,2,3)) + eps
    denominator2 = torch.sum(o2, dim=(1, 2, 3)) + torch.sum(t2, dim=(1, 2, 3)) + eps
    edema_dice = intersect2 / denominator2

    o3 = (output == 3).float()
    t3 = (target == 3).float()
    intersect3 = torch.sum(2 * (o3 * t3), dim=(1,2,3)) + eps
    denominator3 = torch.sum(o3, dim=(1, 2, 3)) + torch.sum(t3, dim=(1, 2, 3)) + eps
    non_enhancing_dice = intersect3 / denominator3

    o4 = (output == 4).float()
    t4 = (target == 4).float()
    intersect4 = torch.sum(2 * (o4 * t4), dim=(1,2,3)) + eps
    denominator4 = torch.sum(o4, dim=(1, 2, 3)) + torch.sum(t4, dim=(1, 2, 3)) + eps
    enhancing_dice = intersect4 / denominator4

    ####post processing:
    if torch.sum(o4) < 500:
        o5 = o4 * 0
    else:
        o5 = o4
    t5 = t4
    intersect5 = torch.sum(2 * (o5 * t5), dim=(1, 2, 3)) + eps
    denominator5 = torch.sum(o5, dim=(1, 2, 3)) + torch.sum(t5, dim=(1, 2, 3)) + eps
    enhancing_dice_postpro = intersect5 / denominator5

    o_whole = o1 + o2 + o3 + o4
    t_whole = t1 + t2 + t3 + t4
    intersect_whole = torch.sum(2 * (o_whole * t_whole), dim=(1,2,3)) + eps
    denominator_whole = torch.sum(o_whole, dim=(1, 2, 3)) + torch.sum(t_whole, dim=(1, 2, 3)) + eps
    dice_whole = intersect_whole / denominator_whole

    o_core = o1 + o3 + o4
    t_core = t1 + t3 + t4
    intersect_core = torch.sum(2 * (o_core * t_core), dim=(1, 2, 3)) + eps
    denominator_core = torch.sum(o_core, dim=(1, 2, 3)) + torch.sum(t_core, dim=(1, 2, 3)) + eps
    dice_core = intersect_core / denominator_core

    dice_evaluate = torch.cat((torch.unsqueeze(dice_whole, 1), torch.unsqueeze(dice_core, 1), torch.unsqueeze(enhancing_dice, 1), torch.unsqueeze(enhancing_dice_postpro, 1)), dim=1)

    return dice_evaluate.cpu().numpy()


def test_softmax(
        test_loader,
        model,
        feature_mask=None,
        dataname='BRATS2020',
        mask_name=None):
    H, W, T = 240, 240, 155
    vals_evaluation = AverageMeter()
    one_tensor = torch.ones(1, patch_size1, patch_size1, patch_size1).float().cuda()

    if dataname in ['BRATS2021', 'BRATS2020', 'BraTS2018']:
        num_cls = 4
        class_evaluation = 'whole', 'core', 'enhancing', 'enhancing_postpro'
        class_separate = 'ncr_net', 'edema', 'enhancing'
    elif dataname == 'BRATS2015':
        num_cls = 5
        class_evaluation = 'whole', 'core', 'enhancing', 'enhancing_postpro'
        class_separate = 'necrosis', 'edema', 'non_enhancing', 'enhancing'

    for i, data in enumerate(test_loader):
        target = data['label'].cuda()
        x = data['image'].cuda()
        names = data['idx']
        if feature_mask is not None:
            mask = torch.from_numpy(np.array(feature_mask))
            mask = torch.unsqueeze(mask, dim=0).repeat(len(names), 1)
        else:
            mask = data[2]
        mask = mask.cuda()

        _, _, H, W, Z = x.size()
        #########get h_ind, w_ind, z_ind for sliding windows
        h_cnt = int(np.ceil((H - patch_size1) / (patch_size1 * (1 - 0.5))))
        h_idx_list = range(0, h_cnt)
        h_idx_list = [h_idx * int(patch_size1 * (1 - 0.5)) for h_idx in h_idx_list]
        h_idx_list.append(H - patch_size1)

        w_cnt = int(np.ceil((W - patch_size1) / (patch_size1 * (1 - 0.5))))
        w_idx_list = range(0, w_cnt)
        w_idx_list = [w_idx * int(patch_size1 * (1 - 0.5)) for w_idx in w_idx_list]
        w_idx_list.append(W - patch_size1)

        z_cnt = int(np.ceil((Z - patch_size1) / (patch_size1 * (1 - 0.5))))
        z_idx_list = range(0, z_cnt)
        z_idx_list = [z_idx * int(patch_size1 * (1 - 0.5)) for z_idx in z_idx_list]
        z_idx_list.append(Z - patch_size1)

        #####compute calculation times for each pixel in sliding windows
        weight1 = torch.zeros(1, 1, H, W, Z).float().cuda()
        for h in h_idx_list:
            for w in w_idx_list:
                for z in z_idx_list:
                    weight1[:, :, h:h + patch_size1, w:w + patch_size1, z:z + patch_size1] += one_tensor
        weight = weight1.repeat(len(names), num_cls, 1, 1, 1)

        #####evaluation
        pred = torch.zeros(len(names), num_cls, H, W, Z).float().cuda()
        model.module.is_training = False
        for h in h_idx_list:
            for w in w_idx_list:
                for z in z_idx_list:
                    x_input = x[:, :, h:h + patch_size1, w:w + patch_size1, z:z + patch_size1]
                    pred_part = model(x_input, mask)[0]
                    pred_part = F.softmax(pred_part, dim=1)
                    pred[:, :, h:h + patch_size1, w:w + patch_size1, z:z + patch_size1] += pred_part
        pred = pred / weight
        b = time.time()
        pred = pred[:, :, :H, :W, :T]
        pred = torch.argmax(pred, dim=1)

        if dataname in ['BRATS2021', 'BRATS2020', 'BraTS2018']:
            scores_evaluation = softmax_output_dice_class4(pred, target)
        elif dataname == 'BRATS2015':
            scores_separate, scores_evaluation = softmax_output_dice_class5(pred, target)
        for k, name in enumerate(names):
            msg = 'Subject {}/{}, {}/{}'.format((i + 1), len(test_loader), (k + 1), len(names))
            msg += '{:>20}, '.format(name)

            vals_evaluation.update(scores_evaluation[k])
            msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_evaluation, scores_evaluation[k])])
            # msg += ',' + ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_separate, scores_separate[k])])

            logging.info(msg)
    msg = '{}_Average scores:'.format(mask_name)
    msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_evaluation, vals_evaluation.avg)])
    logging.info(msg)
    # msg += ',' + ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_separate, vals_evaluation.avg)])
    # print(msg)



    return vals_evaluation.avg

def test_dice_hd95_softmax(
        test_loader,
        model,
        feature_mask=None,
        dataname='BRATS2020',
        mask_name=None,
        csv_name=None,):
    H, W, T = 240, 240, 155
    vals_dice_evaluation = AverageMeter()
    vals_hd95_evaluation = AverageMeter()
    vals_separate = AverageMeter()
    one_tensor = torch.ones(1, patch_size1, patch_size1, patch_size1).float().cuda()

    if dataname in ['BraTS2021', 'BraTS2020', 'BraTS2018']:
        num_cls = 4
        class_evaluation = 'whole', 'core', 'enhancing', 'enhancing_postpro'
        class_separate = 'ncr_net', 'edema', 'enhancing'
    elif dataname == 'BRATS2015':
        num_cls = 5
        class_evaluation = 'whole', 'core', 'enhancing', 'enhancing_postpro'
        class_separate = 'necrosis', 'edema', 'non_enhancing', 'enhancing'

    for i, data in enumerate(test_loader):
        target = data['label'].cuda()
        x = data['image'].cuda()
        names = data['idx']
        if feature_mask is not None:
            mask = torch.from_numpy(np.array(feature_mask))
            mask = torch.unsqueeze(mask, dim=0).repeat(len(names), 1)
        else:
            mask = data[2]
        mask = mask.cuda()

        _, _, H, W, Z = x.size()
        #########get h_ind, w_ind, z_ind for sliding windows
        h_cnt = int(np.ceil((H - patch_size1) / (patch_size1 * (1 - 0.5))))
        h_idx_list = range(0, h_cnt)
        h_idx_list = [h_idx * int(patch_size1 * (1 - 0.5)) for h_idx in h_idx_list]
        h_idx_list.append(H - patch_size1)

        w_cnt = int(np.ceil((W - patch_size1) / (patch_size1 * (1 - 0.5))))
        w_idx_list = range(0, w_cnt)
        w_idx_list = [w_idx * int(patch_size1 * (1 - 0.5)) for w_idx in w_idx_list]
        w_idx_list.append(W - patch_size1)

        z_cnt = int(np.ceil((Z - patch_size1) / (patch_size1 * (1 - 0.5))))
        z_idx_list = range(0, z_cnt)
        z_idx_list = [z_idx * int(patch_size1 * (1 - 0.5)) for z_idx in z_idx_list]
        z_idx_list.append(Z - patch_size1)

        #####compute calculation times for each pixel in sliding windows
        weight1 = torch.zeros(1, 1, H, W, Z).float().cuda()


        for h in h_idx_list:
            for w in w_idx_list:
                for z in z_idx_list:
                    weight1[:, :, h:h + patch_size1, w:w + patch_size1, z:z + patch_size1] += one_tensor
        weight = weight1.repeat(len(names), num_cls, 1, 1, 1)

        #####evaluation
        pred = torch.zeros(len(names), num_cls, H, W, Z).float().cuda()

        model.module.is_training = False
        for h in h_idx_list:
            for w in w_idx_list:
                for z in z_idx_list:
                    x_input = x[:, :, h:h + patch_size1, w:w + patch_size1, z:z + patch_size1]
                    pred_part = model(x_input, mask)[0]
                    pred_part = F.softmax(pred_part, dim=1)
                    pred[:, :, h:h + patch_size1, w:w + patch_size1, z:z + patch_size1] += pred_part
        pred = pred / weight
        b = time.time()
        pred = pred[:, :, :H, :W, :T]
        pred = torch.argmax(pred, dim=1)

        if dataname in ['BraTS2021', 'BraTS2020', 'BraTS2018']:
            scores_separate, scores_evaluation = softmax_output_dice_class4(pred, target)
            scores_hd95 = np.array(cal_hd95(pred[0].cpu().numpy(), target[0].cpu().numpy()))

        elif dataname == 'BRATS2015':
            scores_separate, scores_evaluation = softmax_output_dice_class5(pred, target)
        for k, name in enumerate(names):
            msg = 'Subject {}/{}, {}/{}'.format((i + 1), len(test_loader), (k + 1), len(names))
            msg += '{:>20}, '.format(name)

            vals_separate.update(scores_separate[k])
            vals_dice_evaluation.update(scores_evaluation[k])
            vals_hd95_evaluation.update(scores_hd95)

            msg += 'DSC: '
            msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_evaluation, scores_evaluation[k])])
            # msg += '\nHD95: '
            # msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_evaluation, scores_hd95)])
            file = open(csv_name, "a+")
            csv_writer = csv.writer(file)
            csv_writer.writerow([scores_evaluation[k][0], scores_evaluation[k][1], scores_evaluation[k][2],\
                scores_hd95[0], scores_hd95[1], scores_hd95[2], scores_hd95[3]])
            file.close()
            # msg += ',' + ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_separate, scores_separate[k])])

            logging.info(msg)
    msg = '{}_Average Dice Scores:'.format(mask_name)
    msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_evaluation, vals_dice_evaluation.avg)])
    logging.info(msg)

    # msg = '{}_Average HD95 Scores:'.format(mask_name)
    # msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_evaluation, vals_hd95_evaluation.avg)])
    # logging.info(msg)
    # msg += ',' + ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_separate, vals_evaluation.avg)])
    # print(msg)



    return vals_dice_evaluation.avg, vals_hd95_evaluation.avg

def val_dice_hd95_softmax_BraTS(
        val_loader,
        model,
        feature_mask=None,
        dataname='BRATS2020',
        mask_name=None,
        csv_name=None,):
    H, W, T = 240, 240, 155
    vals_dice_evaluation = AverageMeter()
    vals_hd95_evaluation = AverageMeter()
    vals_separate = AverageMeter()
    one_tensor = torch.ones(1, patch_size1, patch_size1, patch_size1).float().cuda()

    if dataname in ['BraTS2019', 'BraTS2020', 'BraTS2018']:
        num_cls = 4
        class_evaluation = 'whole', 'core', 'enhancing', 'enhancing_postpro'
        class_separate = 'ncr_net', 'edema', 'enhancing'
    elif dataname == 'BRATS2015':
        num_cls = 5
        class_evaluation = 'whole', 'core', 'enhancing', 'enhancing_postpro'
        class_separate = 'necrosis', 'edema', 'non_enhancing', 'enhancing'

    for i, data in enumerate(val_loader):
        target = data['label'].cuda()
        x = data['image'].cuda()
        names = data['idx']
        if feature_mask is not None:
            mask = torch.from_numpy(np.array(feature_mask))
            mask = torch.unsqueeze(mask, dim=0).repeat(len(names), 1)
        else:
            mask = data[2]
        mask = mask.cuda()

        _, _, H, W, Z = x.size()
        #########get h_ind, w_ind, z_ind for sliding windows
        h_cnt = int(np.ceil((H - patch_size1) / (patch_size1 * (1 - 0.5))))
        h_idx_list = range(0, h_cnt)
        h_idx_list = [h_idx * int(patch_size1 * (1 - 0.5)) for h_idx in h_idx_list]
        h_idx_list.append(H - patch_size1)

        w_cnt = int(np.ceil((W - patch_size1) / (patch_size1 * (1 - 0.5))))
        w_idx_list = range(0, w_cnt)
        w_idx_list = [w_idx * int(patch_size1 * (1 - 0.5)) for w_idx in w_idx_list]
        w_idx_list.append(W - patch_size1)

        z_cnt = int(np.ceil((Z - patch_size1) / (patch_size1 * (1 - 0.5))))
        z_idx_list = range(0, z_cnt)
        z_idx_list = [z_idx * int(patch_size1 * (1 - 0.5)) for z_idx in z_idx_list]
        z_idx_list.append(Z - patch_size1)

        #####compute calculation times for each pixel in sliding windows
        weight1 = torch.zeros(1, 1, H, W, Z).float().cuda()


        for h in h_idx_list:
            for w in w_idx_list:
                for z in z_idx_list:
                    weight1[:, :, h:h + patch_size1, w:w + patch_size1, z:z + patch_size1] += one_tensor
        weight = weight1.repeat(len(names), num_cls, 1, 1, 1)

        #####evaluation
        pred = torch.zeros(len(names), num_cls, H, W, Z).float().cuda()

        model.module.is_training = False
        for h in h_idx_list:
            for w in w_idx_list:
                for z in z_idx_list:
                    x_input = x[:, :, h:h + patch_size1, w:w + patch_size1, z:z + patch_size1]
                    pred_part1 = model(x_input, mask)[0]
                    pred_part2 = model(x_input, mask)[1]

                    pred_part1 = F.softmax(pred_part1, dim=1)
                    pred_part2 = F.softmax(pred_part2, dim=1)

                    pred_part = (pred_part1 + pred_part2) / 2.0
                    pred[:, :, h:h + patch_size1, w:w + patch_size1, z:z + patch_size1] += pred_part
        pred = pred / weight
        b = time.time()
        pred = pred[:, :, :H, :W, :T]
        pred = torch.argmax(pred, dim=1)

        if dataname in ['BraTS2019', 'BraTS2020', 'BraTS2018']:
            scores_separate, scores_evaluation = softmax_output_dice_class4(pred, target)
            scores_hd95 = np.array(cal_hd95(pred[0].cpu().numpy(), target[0].cpu().numpy()))

        elif dataname == 'BRATS2015':
            scores_separate, scores_evaluation = softmax_output_dice_class5(pred, target)
        for k, name in enumerate(names):
            msg = 'Subject {}/{}, {}/{}'.format((i + 1), len(val_loader), (k + 1), len(names))
            msg += '{:>20}, '.format(name)

            vals_separate.update(scores_separate[k])
            vals_dice_evaluation.update(scores_evaluation[k])
            vals_hd95_evaluation.update(scores_hd95)

            msg += 'DSC: '
            msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_evaluation, scores_evaluation[k])])
            # msg += '\nHD95: '
            # msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_evaluation, scores_hd95)])
            file = open(csv_name, "a+")
            csv_writer = csv.writer(file)
            csv_writer.writerow([scores_evaluation[k][0], scores_evaluation[k][1], scores_evaluation[k][2],\
                scores_hd95[0], scores_hd95[1], scores_hd95[2]])
            file.close()
            # msg += ',' + ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_separate, scores_separate[k])])

            logging.info(msg)
    msg = '{}_Average Dice Scores:'.format(mask_name)
    msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_evaluation, vals_dice_evaluation.avg)])
    logging.info(msg)

    msg = '{}_Average HD95 Scores:'.format(mask_name)
    msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_evaluation, vals_hd95_evaluation.avg)])
    logging.info(msg)
    # msg += ',' + ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_separate, vals_evaluation.avg)])
    # print(msg)



    return vals_dice_evaluation.avg, vals_hd95_evaluation.avg

def val_dice_hd95_softmax_MyoPS(
        val_loader,
        model,
        feature_mask=None,
        dataname='BRATS2020',
        mask_name=None,
        csv_name=None,):
    H, W, T = 240, 240, 155
    vals_dice_evaluation = AverageMeter()
    vals_hd95_evaluation = AverageMeter()
    vals_separate = AverageMeter()
    one_tensor = torch.ones(1, patch_size_xy, patch_size_xy, patch_size_z).float().cuda()

    if dataname in ['BraTS2019', 'BraTS2020', 'BraTS2018']:
        num_cls = 4
        class_evaluation = 'whole', 'core', 'enhancing', 'enhancing_postpro'
        class_separate = 'ncr_net', 'edema', 'enhancing'
    elif dataname == 'BRATS2015':
        num_cls = 5
        class_evaluation = 'whole', 'core', 'enhancing', 'enhancing_postpro'
        class_separate = 'necrosis', 'edema', 'non_enhancing', 'enhancing'
    elif dataname == 'MyoPS2020':
        num_cls = 6
        class_evaluation = 'Myo', 'MI', 'MI+ME', 'LV', 'RV'

    for i, data in enumerate(val_loader):
        target = data['label'].cuda()
        x = data['image'].cuda()
        names = data['idx']
        if feature_mask is not None:
            mask = torch.from_numpy(np.array(feature_mask))
            mask = torch.unsqueeze(mask, dim=0).repeat(len(names), 1)
        else:
            mask = data[2]
        mask = mask.cuda()

        _, _, H, W, Z = x.size()
        #########get h_ind, w_ind, z_ind for sliding windows
        h_cnt = int(np.ceil((H - patch_size_xy) / (patch_size_xy * (1 - 0.5))))
        h_idx_list = range(0, h_cnt)
        h_idx_list = [h_idx * int(patch_size_xy * (1 - 0.5)) for h_idx in h_idx_list]
        h_idx_list.append(H - patch_size_xy)

        w_cnt = int(np.ceil((W - patch_size_xy) / (patch_size_xy * (1 - 0.5))))
        w_idx_list = range(0, w_cnt)
        w_idx_list = [w_idx * int(patch_size_xy * (1 - 0.5)) for w_idx in w_idx_list]
        w_idx_list.append(W - patch_size_xy)

        z_cnt = int(np.ceil((Z - patch_size_z) / (patch_size_z * (1 - 0.5))))
        z_idx_list = range(0, z_cnt)
        z_idx_list = [z_idx * int(patch_size_z * (1 - 0.5)) for z_idx in z_idx_list]
        z_idx_list.append(Z - patch_size_z)

        #####compute calculation times for each pixel in sliding windows
        weight1 = torch.zeros(1, 1, H, W, Z).float().cuda()


        for h in h_idx_list:
            for w in w_idx_list:
                for z in z_idx_list:
                    weight1[:, :, h:h + patch_size_xy, w:w + patch_size_xy, z:z + patch_size_z] += one_tensor
        weight = weight1.repeat(len(names), num_cls, 1, 1, 1)

        #####evaluation
        pred = torch.zeros(len(names), num_cls, H, W, Z).float().cuda()

        model.module.is_training = False
        for h in h_idx_list:
            for w in w_idx_list:
                for z in z_idx_list:
                    x_input = x[:, :, h:h + patch_size_xy, w:w + patch_size_xy, z:z + patch_size_z]
                    pred_part1 = model(x_input, mask)[0]
                    pred_part2 = model(x_input, mask)[1]

                    pred_part1 = F.softmax(pred_part1, dim=1)
                    pred_part2 = F.softmax(pred_part2, dim=1)

                    pred_part = (pred_part1 + pred_part2) / 2.0
                    pred[:, :, h:h + patch_size_xy, w:w + patch_size_xy, z:z + patch_size_z] += pred_part
        pred = pred / weight
        b = time.time()
        pred = pred[:, :, :H, :W, :T]
        pred = torch.argmax(pred, dim=1)

        if dataname in ['BraTS2019', 'BraTS2020', 'BraTS2018']:
            scores_separate, scores_evaluation = softmax_output_dice_class4(pred, target)
            scores_hd95 = np.array(cal_hd95(pred[0].cpu().numpy(), target[0].cpu().numpy()))

        elif dataname == 'BRATS2015':
            scores_separate, scores_evaluation = softmax_output_dice_class5(pred, target)
        elif dataname == 'MyoPS2020':
            scores_separate, scores_evaluation = softmax_output_dice_MyoPS(pred, target)
            scores_hd95 = np.array(cal_hd95_MyoPS(pred[0].cpu().numpy(), target[0].cpu().numpy()))
        for k, name in enumerate(names):
            msg = 'Subject {}/{}, {}/{}'.format((i + 1), len(val_loader), (k + 1), len(names))
            msg += '{:>20}, '.format(name)

            # vals_separate.update(scores_separate[k])
            vals_dice_evaluation.update(scores_evaluation[k])
            vals_hd95_evaluation.update(scores_hd95)

            msg += 'DSC: '
            msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_evaluation, scores_evaluation[k])])
            # msg += 'HD95: '
            # msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_evaluation, scores_hd95)])
            file = open(csv_name, "a+")
            csv_writer = csv.writer(file)
            csv_writer.writerow([scores_evaluation[k][0], scores_evaluation[k][1], scores_evaluation[k][2],scores_evaluation[k][3],scores_evaluation[k][4], \
                scores_hd95[0], scores_hd95[1], scores_hd95[2], scores_hd95[3], scores_hd95[4]])
            file.close()
            # msg += ',' + ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_separate, scores_separate[k])])

            logging.info(msg)
    msg = '{}_Average Dice Scores:'.format(mask_name)
    msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_evaluation, vals_dice_evaluation.avg)])
    logging.info(msg)

    msg = '{}_Average HD95 Scores:'.format(mask_name)
    msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_evaluation, vals_hd95_evaluation.avg)])
    logging.info(msg)
    # msg += ',' + ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_separate, vals_evaluation.avg)])
    # print(msg)



    return vals_dice_evaluation.avg, vals_hd95_evaluation.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count