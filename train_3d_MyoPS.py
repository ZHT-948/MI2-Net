import csv
import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms

import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from utils.predict_BraTS_MyoPS import AverageMeter

from utils import ramps, losses, metrics, test_patch, predict_BraTS_MyoPS
from dataloaders.dataset import *
from dataloaders.transforms import *

from networks import net_factory


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def sharpen_multi_class(P):
    P = P ** (1 / args.temperature)
    P = P / (P.sum(dim=1, keepdim=True))
    return P

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='MyoPS2020', help='dataset_name')
parser.add_argument('--root_path', type=str, default='/data1/zht/SeSuInMo_Net_main/data/MyoPS2020_tar',
                    help='Name of Dataset')
parser.add_argument('--exp', type=str, default='MI2Net_MyoPS', help='exp_name')
parser.add_argument('--model', type=str, default='MI2Net_MyoPS', help='model_name')
parser.add_argument('--max_iteration', type=int, default=20000, help='maximum iteration to train')
parser.add_argument('--max_samples', type=int, default=32, help='maximum samples to train')
parser.add_argument('--labeled_bs', type=int, default=1, help='batch_size of labeled data per gpu')
parser.add_argument('--batch_size', type=int, default=2, help='batch_size of labeled data per gpu')
parser.add_argument('--base_lr', type=float, default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--labelnum', type=int, default=16, help='trained samples')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--gpu', type=str, default='2,3', help='GPU to use')
parser.add_argument('--consistency', type=float, default=1, help='consistency_weight')
parser.add_argument('--consistency_rampup', type=float, default=40.0, help='consistency_rampup')
parser.add_argument('--temperature', type=float, default=0.1, help='temperature of sharpening')
parser.add_argument('--lamda', type=float, default=0.5, help='weight to balance all losses')
parser.add_argument('--region_fusion_start_epoch', type=int, default=0)
parser.add_argument('--csvpath', type=str, default='/home/zht/Code/MI2-Net/code/csv/')
parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--resume_iter', type=int, default=None)
parser.add_argument('--resume_dice', type=float, default=None)
args = parser.parse_args()

snapshot_path = "/data2/zht/MI2-Net/model_final/{}_{}_{}_labeled_{}_data_{}/{}/train".format(
    args.dataset_name, args.exp, args.max_iteration, args.batch_size, args.max_samples, args.labelnum)
csvpath = "/data2/zht/MI2-Net/csv_final/{}_{}_{}_labeled/{}/train".format(args.dataset_name, args.exp,
                                                                                         args.max_iteration,
                                                                                         args.labelnum, args.model)
num_classes = 6
if args.dataset_name == "MyoPS2020":
    patch_size = (288, 288, 16)
    args.root_path = args.root_path
    args.max_samples = args.max_samples

train_data_path = args.root_path

# modality missing mask
if args.dataset_name == 'MyoPS2020':
    masks = [[False, False, True], [False, True, False], [True, False, False],
             [True, True, False],  [True, False, True], [False, True, True],
             [True, True, True]]
    mask_name = ['T2', 'LGE', 'bssFP',
                 'LGE_bssFP',  'T2_bssFP', 'T2_LGE',
                 'T2_LGE_bssFP']

masks_torch = torch.from_numpy(np.array(masks))
print(masks_torch.int())

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
print(torch.cuda.device_count())

csvpath = csvpath
os.makedirs(csvpath, exist_ok=True)

labeled_bs = args.labeled_bs
max_iterations = args.max_iteration
base_lr = args.base_lr
args.train_transforms = 'Compose([RandCrop3D((288,288,16)), RandomRotion(10), RandomIntensityChange((0.1,0.1)), RandomFlip(0), NumpyType((np.float32, np.int64)),])'
args.test_transforms = 'Compose([NumpyType((np.float32, np.int64)),])'

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

if __name__ == "__main__":
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    model = net_factory.net_factorys(net_type=args.model, in_chns=3, class_num=num_classes, mode="train")
    model = torch.nn.DataParallel(model).cuda()


    if args.dataset_name == 'MyoPS2020':
        train_file = '/home/zht/Code/MI2-Net-main/MyoPS/MyoPS2020_Training_none_npy/train.txt'
        test_file = '/home/zht/Code/MI2-Net-main/MyoPS/MyoPS2020_Training_none_npy/test.txt'
        valid_file = '/home/zht/Code/MI2-Net-main/MyoPS/MyoPS2020_Training_none_npy/valid.txt'

    if args.dataset_name == "MyoPS2020":
        db_train = Train_MyoPSDataSets(base_dir=train_data_path,
                                       num_cls=num_classes,
                                       transform=args.train_transforms,
                                       train_file=train_file
                                       )
        db_valid = Test_MyoPSDataSets(base_dir=train_data_path,
                                      num_cls=num_classes,
                                      transform=args.test_transforms,
                                      train_file=valid_file)
    elif args.dataset_name == "Pancreas_CT":
        db_train = Pancreas(base_dir=train_data_path,
                            split='train',
                            transform=transforms.Compose([
                                RandomCrop(patch_size),
                                ToTensor(),
                            ]))
    all_idxs = list(range(args.max_samples))

    rng = random.Random(args.seed)
    rng.shuffle(all_idxs)

    labeled_idxs = all_idxs[:args.labelnum]
    unlabeled_idxs = all_idxs[args.labelnum:]
    print("========================== Labeled samples =========================")
    print(f"✅ Total labeled = {len(labeled_idxs)} ")
    for index in labeled_idxs:
        case_name = db_train.names[index]
        print(case_name)
    batch_sampler = TwoStreamBatchSampler(labeled_idxs,
                                          unlabeled_idxs,
                                          args.batch_size,
                                          args.batch_size - labeled_bs)


    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)


    trainloader = DataLoader(
        db_train,
        batch_sampler=batch_sampler,
        num_workers=8,
        pin_memory=True,
        worker_init_fn=worker_init_fn)
    validloader = DataLoader(
        db_valid,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))
    consistency_criterion = losses.MSE_loss
    dice_loss = losses.Binary_dice_loss
    iter_num = 0
    best_dice = 0
    if args.resume:
        if os.path.isfile(args.resume):
            logging.info(f"Loading checkpoint '{args.resume}'")

            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint)

            if args.resume_iter > 0:
                iter_num = args.resume_iter
                logging.info(f"Manually set iter_num to {iter_num}")

            if args.resume_dice > 0:
                best_dice = args.resume_dice
                logging.info(f"Manually set best_dice to {best_dice}")

            logging.info("Resuming training successfully!")
        else:
            logging.info(f"No checkpoint found at '{args.resume}'")
    loss_min = 1
    max_epoch = max_iterations // len(trainloader) + 1
    lr_ = base_lr
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch[0]['image'], sampled_batch[0]['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            mask = sampled_batch[1]
            mask = mask.cuda()

            model.train()
            model.module.is_training = True
            outputs_fuse, outputs_sep,  outputs_preds, rec = model(volume_batch, mask)

            num_outputs_fuse = len(outputs_fuse)
            num_outputs_sep = len(outputs_sep)
            num_outputs_preds = len(outputs_preds[0])


            loss_seg_cross_fuse = 0
            loss_seg_dice_fuse = 0

            loss_seg_cross_preds = 0
            loss_seg_dice_preds = 0

            y_ori_fuse = []
            y_pseudo_label_fuse = []

            for idx in range(num_outputs_fuse):
                y_fuse = outputs_fuse[idx][:labeled_bs, ...]
                y_prob_fuse = F.softmax(y_fuse, dim=1)
                loss_seg_cross_fuse += losses.softmax_weighted_loss(y_prob_fuse, label_batch[:labeled_bs],
                                                                    numcls=num_classes)
                loss_seg_dice_fuse += losses.dice_loss(y_prob_fuse, label_batch[:labeled_bs], num_cls=num_classes)

                y_all_fuse = outputs_fuse[idx]
                y_prob_all_fuse = F.softmax(y_all_fuse, dim=1)
                y_ori_fuse.append(y_prob_all_fuse)
                y_pseudo_label_fuse.append(sharpen_multi_class(y_prob_all_fuse).detach())

            for jdx in range(num_outputs_fuse):
                for kdx in range(num_outputs_preds):
                    y_preds = outputs_preds[jdx][kdx]

                    y_preds_labeled = y_preds[:labeled_bs, ...]

                    y_prob_preds = F.softmax(y_preds_labeled, dim=1)

                    loss_seg_cross_preds += losses.softmax_weighted_loss(y_prob_preds, label_batch[:labeled_bs],
                                                                         numcls=num_classes)
                    loss_seg_dice_preds += losses.dice_loss(y_prob_preds, label_batch[:labeled_bs], num_cls=num_classes)

            loss_fuse = loss_seg_cross_fuse + loss_seg_dice_fuse + loss_seg_cross_preds + loss_seg_dice_preds

            loss_consist_fuse = 0
            for i in range(num_outputs_fuse):
                for j in range(num_outputs_fuse):
                    if i != j:
                        loss_consist_fuse += consistency_criterion(y_ori_fuse[i], y_pseudo_label_fuse[j], num_cls=num_classes)


            teacher_soft = y_pseudo_label_fuse[0].detach()

            loss_seg_cross_sep = 0
            loss_seg_dice_sep = 0
            loss_consist_sep = 0

            for idx in range(num_outputs_sep):
                y_sep = outputs_sep[idx][:labeled_bs, ...]
                y_prob_sep = F.softmax(y_sep, dim=1)
                loss_seg_cross_sep += losses.softmax_weighted_loss(y_prob_sep, label_batch[:labeled_bs,],
                                                                   numcls=num_classes)
                loss_seg_dice_sep += losses.dice_loss(y_prob_sep, label_batch[:labeled_bs,], num_cls=num_classes)

                y_all_sep = outputs_sep[idx]
                y_prob_all_sep = F.softmax(y_all_sep, dim=1)

                loss_consist_sep += consistency_criterion(y_prob_all_sep, teacher_soft, num_cls=num_classes)


            loss_sep = loss_seg_cross_sep + loss_seg_dice_sep

            rec_list, gt_list = rec

            loss_rec = 0
            beta = 0.01
            for i in range(len(rec_list)):
                rec = rec_list[i]
                gt = gt_list[i].detach()

                rec_norm = F.instance_norm(rec)
                gt_norm = F.instance_norm(gt)

                mask_broadcast = mask.view(rec.shape[0], 3, 1, 1, 1, 1).type_as(rec)

                loss_map = F.mse_loss(rec_norm, gt_norm, reduction='none')

                mask_weight = 1 - mask_broadcast
                elements_per_modality = rec.shape[2] * rec.shape[3] * rec.shape[4] * rec.shape[5]

                loss_scale_i = (loss_map * mask_weight).sum() / (
                        mask_weight.sum() * elements_per_modality + 1e-6)

                loss_rec += loss_scale_i

            iter_num = iter_num + 1
            consistency_weight = get_current_consistency_weight(iter_num // 200)

            loss_consist = loss_consist_fuse + loss_consist_sep

            if iter_num < args.region_fusion_start_epoch:
                loss = args.lamda * (loss_fuse * 0.0 + loss_sep) + beta * loss_rec + consistency_weight * loss_consist
            else:
                loss = args.lamda * (loss_fuse + loss_sep) + beta * loss_rec + consistency_weight * loss_consist
            # loss = args.lamda * loss_seg_dice + consistency_weight * loss_consist
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logging.info('iteration %d : loss : %03f, '
                         'loss_cosist: %03f, '
                         'loss_d_f: %03f, '
                         'loss_c_f: %03f, '
                         'loss_d_s: %03f, '
                         'loss_c_s: %03f, '
                         'loss_d_p: %03f, '
                         'loss_c_p: %03f, '
                         'loss_rec: %03f, ' % (iter_num, loss, loss_consist, loss_seg_dice_fuse, loss_seg_cross_fuse,
                                               loss_seg_dice_sep, loss_seg_cross_sep, loss_seg_dice_preds,
                                               loss_seg_cross_preds, loss_rec))

            writer.add_scalar('Labeled_loss_fuse/loss_seg_dice_fuse', loss_seg_dice_fuse, iter_num)
            writer.add_scalar('Labeled_loss_fuse/loss_seg_ce_fuse', loss_seg_cross_fuse, iter_num)
            writer.add_scalar('Labeled_loss_preds/loss_seg_dice_preds', loss_seg_dice_preds, iter_num)
            writer.add_scalar('Labeled_loss_preds/loss_seg_ce_preds', loss_seg_cross_preds, iter_num)
            writer.add_scalar('Labeled_loss_sep/loss_seg_dice_sep', loss_seg_dice_sep, iter_num)
            writer.add_scalar('Labeled_loss_sep/loss_seg_ce_sep', loss_seg_cross_sep, iter_num)
            writer.add_scalar('Co_loss/consistency_loss_fuse', loss_consist_fuse, iter_num)
            writer.add_scalar('Co_loss/consistency_loss_sep', loss_consist_sep, iter_num)
            writer.add_scalar('Co_loss/consistency_loss', loss_consist, iter_num)
            writer.add_scalar('Co_loss/consist_weight', consistency_weight, iter_num)
            writer.add_scalar('rec_loss/rec_loss', loss_rec, iter_num)


            if iter_num >= 1000 and iter_num % 1000 == 0 or (iter_num >= (args.max_iteration - 10)):

                model.eval()
                test_dice_socre = AverageMeter()
                test_hd95_socre = AverageMeter()
                csv_name = os.path.join(csvpath, '{}.csv'.format(args.labelnum))

                with torch.no_grad():
                    logging.info('#################test model#################')
                    file = open(csv_name, "a+")
                    csv_writer = csv.writer(file)
                    csv_writer.writerow(
                        ['Myo Dice', 'MI Dice', 'MI+ME Dice', 'LV Dice', 'RV Dice',
                         'Myo HD95', 'MI HD95', 'MI+ME HD95', 'LV HD95', 'RV HD95'])
                    file.close()


                    if args.dataset_name == "MyoPS2020":
                        for i, mask in enumerate(masks):
                            logging.info('Type name: {}'.format(mask_name[i]))
                            file = open(csv_name, "a+")
                            csv_writer = csv.writer(file)
                            csv_writer.writerow([mask_name[i]])
                            file.close()
                            dice_score, hd95_score = predict_BraTS_MyoPS.val_dice_hd95_softmax_MyoPS(val_loader=validloader,
                                                                                             model=model,
                                                                                             feature_mask=mask,
                                                                                             dataname=args.dataset_name,
                                                                                             mask_name=mask_name[i],
                                                                                             csv_name=csv_name,
                                                                                             )
                            test_dice_socre.update(dice_score)
                            test_hd95_socre.update(hd95_score)

                            dice_score = np.mean(test_dice_socre.avg)

                        logging.info('Avg scores: {}'.format(test_dice_socre.avg))
                        logging.info('Avg scores: {}'.format(test_hd95_socre.avg))

                if dice_score > best_dice:
                    best_dice = dice_score
                    save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}.pth'.format(iter_num, best_dice))
                    save_best_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best_path)
                    logging.info("save best model to {}".format(save_mode_path))

                writer.add_scalar('Var_dice/Dice', dice_score, iter_num)
                writer.add_scalar('Var_dice/Best_dice', best_dice, iter_num)

                model.train()


            if iter_num >= max_iterations:
                save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
