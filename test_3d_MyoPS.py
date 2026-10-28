import os
import argparse
import torch
from utils.predict_BraTS_MyoPS import AverageMeter
from networks import net_factory
from utils.test_patch import test_all_case
import numpy as np
import logging
from utils import ramps, losses, metrics, test_patch, predict_BraTS_MyoPS
from dataloaders.dataset import *
from torch.utils.data import DataLoader
import csv
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str,  default='MyoPS2020', help='dataset_name')
parser.add_argument('--root_path', type=str, default='/data1/zht/data/MyoPS2020_tar', help='Name of Experiment')
parser.add_argument('--exp', type=str,  default='MI2Net_MyoPS', help='exp_name')
parser.add_argument('--model', type=str,  default='MI2Net_MyoPS', help='model_name')
parser.add_argument('--gpu', type=str,  default='3', help='GPU to use')
parser.add_argument('--detail', type=int,  default=1, help='print metrics for every samples?')
parser.add_argument('--labelnum', type=int, default=16, help='labeled data')
parser.add_argument('--max_iteration', type=int, default=20000, help='labeled data')
parser.add_argument('--batch_size', type=int, default=2, help='batch_size of labeled data per gpu')
parser.add_argument('--max_samples', type=int,  default=32, help='maximum samples to train')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
snapshot_path = "/data2/zht/MI2-Net/model_final/{}_{}_{}_labeled_{}_data_{}/{}/train".format(
    args.dataset_name, args.exp, args.max_iteration, args.batch_size, args.max_samples, args.labelnum)
test_save_path = "/data2/zht/MI2-Net/model_final/{}_{}_{}_labeled_{}_data_{}/{}/test".format(
    args.dataset_name, args.exp, args.max_iteration, args.batch_size, args.max_samples, args.labelnum)
args.test_transforms = 'Compose([NumpyType((np.float32, np.int64)),])'
csvpath = ''

num_classes = 6
if args.dataset_name == "MyoPS2020":
    patch_size = (288, 288, 16)
    args.root_path = args.root_path
    args.max_samples = args.max_samples
test_data_path = args.root_path

# modality missing mask

if args.dataset_name == 'MyoPS2020':
    masks = [[False, False, True], [False, True, False], [True, False, False],
             [True, True, False], [True, False, True], [False, True, True],
             [True, True, True]]
    mask_name = ['T2', 'LGE', 'bssFP',
                 'LGE_bssFP', 'T2_bssFP', 'T2_LGE',
                 'T2_LGE_bssFP']


if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)
print(test_save_path)
logging.basicConfig(filename=test_save_path + "/log.txt", level=logging.INFO,
                    format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
logging.info(str(args))

def test_calculate_metric():

    if args.dataset_name == 'MyoPS2020':
        test_file = '/home/zht/Code/MI2-Net-main/MyoPS/MyoPS2020_Training_none_npy/test.txt'


    if args.dataset_name == "MyoPS2020":
        db_test = Test_MyoPSDataSets(base_dir=test_data_path,
                                       num_cls=num_classes,
                                       transform=args.test_transforms,
                                       train_file=test_file
                                       )

        testloader = DataLoader(
        db_test,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
        )


    model = net_factory.net_factorys(net_type=args.model, in_chns=3, class_num=num_classes, mode="test")
    model = torch.nn.DataParallel(model).cuda()
    save_mode_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
    model.load_state_dict(torch.load(save_mode_path), strict=False)
    print("init weight from {}".format(save_mode_path))
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
                dice_score, hd95_score = predict_BraTS_MyoPS.val_dice_hd95_softmax_MyoPS(val_loader=testloader,
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


if __name__ == '__main__':
    metric = test_calculate_metric()
    print(metric)
