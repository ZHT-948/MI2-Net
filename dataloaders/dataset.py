import glob
import os.path

import torch
import numpy as np
from torch.utils.data import Dataset
import h5py
import itertools
from scipy import ndimage
import random
from torch.utils.data.sampler import Sampler
from skimage import transform as sk_trans
from scipy.ndimage import rotate, zoom
from .transforms import *
import torch.distributed as dist

masks_BraTS2018 = np.array([[False, False, False, True], [False, True, False, False], [False, False, True, False],
         [True, False, False, False],
         [False, True, False, True], [False, True, True, False], [True, False, True, False], [False, False, True, True],
         [True, False, False, True], [True, True, False, False],
         [True, True, True, False], [True, False, True, True], [True, True, False, True], [False, True, True, True],
         [True, True, True, True]])

masks_MyoPS2020 = np.array([[False, False, True], [False, True, False], [True, False, False],
             [True, True, False], [True, False, True], [False, True, True],
             [True, True, True]])


class Train_BraTSDataSets(Dataset):
    def __init__(self, base_dir=None, data_path=None, num_cls=4, modal='all', transform='', train_file='train_txt'):
        data_file_path = os.path.join(base_dir, train_file)
        with open(data_file_path, 'r') as f:
            datalist = [i.strip() for i in f.readlines()]
        datalist.sort()

        vol_paths = []
        for dataname in datalist:
            vol_paths.append(os.path.join(base_dir, 'vol', dataname + '_vol.npy'))

        self.vol_paths = vol_paths
        self.transforms = eval(transform or 'Identity()')
        self.names = datalist
        self.num_cls = num_cls



        if modal == 'flair':
            self.modal_ind = np.array([0])
        elif modal == 't1ce':
            self.modal_ind = np.array([1])
        elif modal == 't1':
            self.modal_ind = np.array([2])
        elif modal == 't2':
            self.modal_ind = np.array([3])
        elif modal == 'all':
            self.modal_ind = np.array([0,1,2,3])


    def __getitem__(self, index):

        vol_path = self.vol_paths[index]
        name = self.names[index]


        image = np.load(vol_path)
        seg_path = vol_path.replace('vol', 'seg')
        label = np.load(seg_path)

        image, label = image[None, ...], label[None, ...]  #[B, C, H, W, D]
        image, label = self.transforms([image, label])

        image = np.ascontiguousarray(image.transpose(0, 4, 1, 2, 3))
        _, H, W, Z = np.shape(label)
        label = np.reshape(label, (-1))
        one_hot_targets = np.eye(self.num_cls)[label]

        label = np.reshape(one_hot_targets, (1, H, W, Z, -1))
        label = np.ascontiguousarray(label.transpose(0, 4, 1, 2, 3))

        image = image[:, self.modal_ind, :, :, :]

        image = torch.squeeze(torch.from_numpy(image).float(), dim=0)
        label = torch.squeeze(torch.from_numpy(label).long(), dim=0)

        mask_idx = np.random.choice(15, 1)
        mask = torch.squeeze(torch.from_numpy(masks_BraTS2018[mask_idx]), dim=0)

        sample = {
            'image': image,
            'label': label,
            'idx': name
        }

        return sample, mask


    def __len__(self):
        return len(self.vol_paths)

class Train_MyoPSDataSets(Dataset):
    def __init__(self, base_dir=None, data_path=None, num_cls=4, modal='all', transform='', train_file='train_txt'):
        data_file_path = os.path.join(base_dir, train_file)
        with open(data_file_path, 'r') as f:
            datalist = [i.strip() for i in f.readlines()]
        datalist.sort()

        vol_paths = []
        for dataname in datalist:
            vol_paths.append(os.path.join(base_dir, 'vol', dataname + '_vol.npy'))

        self.vol_paths = vol_paths
        self.transforms = eval(transform or 'Identity()')
        self.names = datalist
        self.num_cls = num_cls



        if modal == 'bssFP':
            self.modal_ind = np.array([0])
        elif modal == 'LGE':
            self.modal_ind = np.array([1])
        elif modal == 'T2':
            self.modal_ind = np.array([2])
        elif modal == 'all':
            self.modal_ind = np.array([0,1,2])


    def __getitem__(self, index):

        vol_path = self.vol_paths[index]
        name = self.names[index]


        image = np.load(vol_path)
        seg_path = vol_path.replace('vol', 'seg')
        label = np.load(seg_path)

        image, label = image[None, ...], label[None, ...]  #[B, C, H, W, D]
        image, label = self.transforms([image, label])

        image = np.ascontiguousarray(image.transpose(0, 4, 1, 2, 3))
        _, H, W, Z = np.shape(label)
        label = np.reshape(label, (-1))
        one_hot_targets = np.eye(self.num_cls)[label]

        label = np.reshape(one_hot_targets, (1, H, W, Z, -1))
        label = np.ascontiguousarray(label.transpose(0, 4, 1, 2, 3))

        image = image[:, self.modal_ind, :, :, :]

        image = torch.squeeze(torch.from_numpy(image).float(), dim=0)
        label = torch.squeeze(torch.from_numpy(label).long(), dim=0)

        mask_idx = np.random.choice(7, 1)
        mask = torch.squeeze(torch.from_numpy(masks_MyoPS2020[mask_idx]), dim=0)

        sample = {
            'image': image,
            'label': label,
            'idx': name
        }

        return sample, mask


    def __len__(self):
        return len(self.vol_paths)


class Test_BraTSDataSets(Dataset):
    def __init__(self, base_dir=None, num_cls=4, modal='all', transform='', train_file='test_txt'):
        data_file_path = os.path.join(base_dir, train_file)
        with open(data_file_path, 'r') as f:
            datalist = [i.strip() for i in f.readlines()]
        datalist.sort()

        vol_paths = []
        for dataname in datalist:
            vol_paths.append(os.path.join(base_dir, 'vol', dataname + '_vol.npy'))

        self.vol_paths = vol_paths
        self.transforms = eval(transform or 'Identity()')
        self.names = datalist
        self.num_cls = num_cls

        if modal == 'flair':
            self.modal_ind = np.array([0])
        elif modal == 't1ce':
            self.modal_ind = np.array([1])
        elif modal == 't1':
            self.modal_ind = np.array([2])
        elif modal == 't2':
            self.modal_ind = np.array([3])
        elif modal == 'all':
            self.modal_ind = np.array([0, 1, 2, 3])

    def __getitem__(self, index):

        vol_path = self.vol_paths[index]
        name = self.names[index]

        image = np.load(vol_path)
        seg_path = vol_path.replace('vol', 'seg')
        label = np.load(seg_path)

        image, label = image[None, ...], label[None, ...]
        image, label = self.transforms([image, label])

        image = np.ascontiguousarray(image.transpose(0, 4, 1, 2, 3))
        label = np.ascontiguousarray(label)

        image = image[:, self.modal_ind, :, :, :]

        image = torch.squeeze(torch.from_numpy(image).float(), dim=0)
        label = torch.squeeze(torch.from_numpy(label).long(), dim=0)



        sample = {
            'image': image,
            'label': label,
            'idx': name
        }

        return sample

    def __len__(self):
        return len(self.vol_paths)

class Test_MyoPSDataSets(Dataset):
    def __init__(self, base_dir=None, num_cls=4, modal='all', transform='', train_file='test_txt'):
        data_file_path = os.path.join(base_dir, train_file)
        with open(data_file_path, 'r') as f:
            datalist = [i.strip() for i in f.readlines()]
        datalist.sort()

        vol_paths = []
        for dataname in datalist:
            vol_paths.append(os.path.join(base_dir, 'vol', dataname + '_vol.npy'))

        self.vol_paths = vol_paths
        self.transforms = eval(transform or 'Identity()')
        self.names = datalist
        self.num_cls = num_cls

        if modal == 'bssFP':
            self.modal_ind = np.array([0])
        elif modal == 'LGE':
            self.modal_ind = np.array([1])
        elif modal == 'T2':
            self.modal_ind = np.array([2])
        elif modal == 'all':
            self.modal_ind = np.array([0, 1, 2])

    def __getitem__(self, index):

        vol_path = self.vol_paths[index]
        name = self.names[index]

        image = np.load(vol_path)
        seg_path = vol_path.replace('vol', 'seg')
        label = np.load(seg_path)

        image, label = image[None, ...], label[None, ...]
        image, label = self.transforms([image, label])

        image = np.ascontiguousarray(image.transpose(0, 4, 1, 2, 3))
        label = np.ascontiguousarray(label)

        image = image[:, self.modal_ind, :, :, :]

        image = torch.squeeze(torch.from_numpy(image).float(), dim=0)
        label = torch.squeeze(torch.from_numpy(label).long(), dim=0)



        sample = {
            'image': image,
            'label': label,
            'idx': name
        }

        return sample

    def __len__(self):
        return len(self.vol_paths)

class BaseDataSets(Dataset):
    def __init__(self, base_dir=None, split='train', num=None, transform=None):
        self._base_dir = base_dir  # 存储数据集路径
        self.sample_list = []  # 初始化一个空列表，用于存储样本名称
        self.split = split  # 记录当前是训练集还是验证集
        self.transform = transform  # 保存数据增强方法

        if self.split == 'train':
            with open(self._base_dir + '/train_slices.list', 'r') as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace('\n', '') for item in self.sample_list]

        elif self.split == 'val':
            with open(self._base_dir + '/val.list', 'r') as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace('\n', '') for item in self.sample_list]
        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self._base_dir + "/data/slices/{}.h5".format(case), 'r')
        else:
            h5f = h5py.File(self._base_dir + "/data/{}.h5".format(case), 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        sample = {'image': image, 'label': label}
        if self.split == "train":
            sample = self.transform(sample)
        sample["idx"] = idx
        return sample

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # ind = random.randrange(0, img.shape[0])
        # image = img[ind, ...]
        # label = lab[ind, ...]
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {'image': image, 'label': label}
        return sample


class LAHeart(Dataset):
    """ LA Dataset """
    def __init__(self, base_dir=None, split='train', num=None, transform=None):
        self._base_dir = base_dir
        self.transform = transform
        self.sample_list = []

        train_path = self._base_dir+'/train.list'
        test_path = self._base_dir+'/test.list'

        if split=='train':
            with open(train_path, 'r') as f:
                self.image_list = f.readlines()
        elif split == 'test':
            with open(test_path, 'r') as f:
                self.image_list = f.readlines()

        self.image_list = [item.replace('\n','') for item in self.image_list]
        if num is not None:
            self.image_list = self.image_list[:num]
        print("total {} samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        h5f = h5py.File(self._base_dir + "/2018LA_Seg_Training Set/" + image_name + "/mri_norm2.h5", 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)

        return sample

class Pancreas(Dataset):
    """ Pancreas Dataset """
    def __init__(self, base_dir=None, split='train', num=None, transform=None):
        self._base_dir = base_dir
        self.transform = transform
        self.sample_list = []

        train_path = self._base_dir+'/train.list'
        test_path = self._base_dir+'/test.list'

        if split=='train':
            with open(train_path, 'r') as f:
                self.image_list = f.readlines()
        elif split == 'test':
            with open(test_path, 'r') as f:
                self.image_list = f.readlines()

        self.image_list = [item.replace('\n','') for item in self.image_list]
        if num is not None:
            self.image_list = self.image_list[:num]
        print("total {} samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        h5f = h5py.File(self._base_dir + "/Pancreas_h5/" + image_name + "_norm.h5", 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)

        return sample

class Resize(object):

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        (w, h, d) = image.shape
        label = label.astype(np.bool)
        image = sk_trans.resize(image, self.output_size, order = 1, mode = 'constant', cval = 0)
        label = sk_trans.resize(label, self.output_size, order = 0)
        assert(np.max(label) == 1 and np.min(label) == 0)
        assert(np.unique(label).shape[0] == 2)
        
        return {'image': image, 'label': label}
    
    
class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape

        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]

        return {'image': image, 'label': label}


class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size, with_sdf=False):
        self.output_size = output_size
        self.with_sdf = with_sdf

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if self.with_sdf:
            sdf = sample['sdf']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            if self.with_sdf:
                sdf = np.pad(sdf, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape

        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        if self.with_sdf:
            sdf = sdf[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            return {'image': image, 'label': label, 'sdf': sdf}
        else:
            return {'image': image, 'label': label}


class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image, label = random_rot_flip(image, label)

        return {'image': image, 'label': label}

class RandomRot(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image, label = random_rotate(image, label)

        return {'image': image, 'label': label}


class RandomNoise(object):
    def __init__(self, mu=0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        noise = np.clip(self.sigma * np.random.randn(image.shape[0], image.shape[1], image.shape[2]), -2*self.sigma, 2*self.sigma)
        noise = noise + self.mu
        image = image + noise
        return {'image': image, 'label': label}


class CreateOnehotLabel(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        onehot_label = np.zeros((self.num_classes, label.shape[0], label.shape[1], label.shape[2]), dtype=np.float32)
        for i in range(self.num_classes):
            onehot_label[i, :, :, :] = (label == i).astype(np.float32)
        return {'image': image, 'label': label,'onehot_label':onehot_label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        if 'onehot_label' in sample:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long(),
                    'onehot_label': torch.from_numpy(sample['onehot_label']).long()}
        else:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long()}


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)


class DistributedTwoStreamBatchSampler(Sampler):
    """Distributed version of TwoStreamBatchSampler for DDP training

    Each process gets a different subset of the data while maintaining
    the two-stream (labeled + unlabeled) structure for semi-supervised learning.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size,
                 secondary_batch_size, num_replicas=None, rank=None, seed=0):
        """
        Args:
            primary_indices: labeled data indices
            secondary_indices: unlabeled data indices
            batch_size: total batch size per process
            secondary_batch_size: unlabeled batch size per process
            num_replicas: number of processes (usually world_size)
            rank: current process rank
            seed: random seed for reproducibility
        """
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.seed = seed

        # Split primary indices across replicas
        self.primary_indices = self._split_indices(primary_indices, num_replicas, rank)
        # Split secondary indices across replicas
        self.secondary_indices = self._split_indices(secondary_indices, num_replicas, rank)

        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0, \
            f"Primary indices ({len(self.primary_indices)}) < primary batch size ({self.primary_batch_size})"
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0, \
            f"Secondary indices ({len(self.secondary_indices)}) < secondary batch size ({self.secondary_batch_size})"

        # Calculate length based on primary indices for this replica
        self.length = len(self.primary_indices) // self.primary_batch_size

    def _split_indices(self, indices, num_replicas, rank):
        """Split indices across replicas"""
        # Shuffle indices deterministically based on seed
        rng = np.random.RandomState(self.seed)
        shuffled_indices = rng.permutation(indices)

        # Split across replicas
        total_size = len(shuffled_indices)
        num_samples_per_replica = total_size // num_replicas
        start_idx = rank * num_samples_per_replica
        end_idx = start_idx + num_samples_per_replica

        # Last replica gets any remaining samples
        if rank == num_replicas - 1 and end_idx < total_size:
            end_idx = total_size

        return shuffled_indices[start_idx:end_idx].tolist()

    def set_epoch(self, epoch):
        """Set epoch for shuffling - important for DDP"""
        self.epoch = epoch

    def __iter__(self):
        # Use epoch and rank to create unique random state for each process and epoch
        rng = np.random.RandomState(self.seed + self.epoch + self.rank * 1000)

        # Shuffle indices for this epoch
        primary_shuffled = rng.permutation(self.primary_indices).tolist()
        secondary_shuffled = rng.permutation(self.secondary_indices).tolist()

        # Create iterators
        primary_iter = self._iterate_once(primary_shuffled)
        secondary_iter = self._iterate_eternally(secondary_shuffled, rng)

        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(self._grouper(primary_iter, self.primary_batch_size),
                   self._grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return self.length

    def _iterate_once(self, indices):
        """Iterate through indices once in shuffled order"""
        return iter(indices)

    def _iterate_eternally(self, indices, rng):
        """Iterate through indices eternally with different shuffles each time"""

        def infinite_shuffles():
            while True:
                # Use the provided RNG to maintain process-specific shuffling
                yield rng.permutation(indices).tolist()

        return itertools.chain.from_iterable(infinite_shuffles())

    def _grouper(self, iterable, n):
        """Collect data into fixed-length chunks or blocks"""
        args = [iter(iterable)] * n
        return zip(*args)

