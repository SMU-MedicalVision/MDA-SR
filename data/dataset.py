import os
import tqdm
import torch
import random
import pickle
import imageio, glob
import numpy as np
import torch.utils.data as data
from PIL import Image, ImageFile
from torchvision.transforms import Compose, Resize, RandomCrop, CenterCrop, RandomHorizontalFlip, RandomVerticalFlip, ToTensor, Normalize

ImageFile.LOAD_TRUNCATED_IMAGES = True


class unpaired_dataset(data.Dataset):
    def __init__(self, args, phase='train'):
        self.args = args
        self.images_source = glob.glob(os.path.join("/home/ltb/dataset/Gastroscopy_Data/MERGE_ABC", '*.jpg'))
        # self.images_source = glob.glob(os.path.join("/home/ltb/dataset/Gastroscopy_Data/NS-inpainting-deartifact-final-SR", '*.jpg'))
        # self.images_source += glob.glob(os.path.join("/home/ltb/dataset/Gastroscopy_Data/clearn_up-deartifact", "*.jpg"))
        # self.images_source += glob.glob(os.path.join("/home/ltb/dataset/Gastroscopy_Data/gastroscopy_data_NS_deartifact", "*.jpg"))
        self.images_target = glob.glob(os.path.join("/home/ltb/dataset/Capsule_Data/An_Han", '*.*'))
        if phase == 'train':
            self.images_source = self.images_source[:-33]
        else:
            self.images_source = glob.glob(os.path.join("/home/ltb/project/MDA-SR/experiment/dataset/test02", '*.jpg'))#self.images_source[:]
            self.images_target = glob.glob(os.path.join("/home/ltb/dataset/Capsule_Data/TestSet/Capsule_dataset01", '*.*'))

        self.phase = phase
        self.binary = False

        print('\nphase: {}'.format(phase))

        self.images_source_size = len(self.images_source)
        self.images_target_size = len(self.images_target)

        patches_source_size = len(self.images_source)
        patches_target_size = len(self.images_target)

        self.dataset_size = int(min(patches_source_size, patches_target_size))  # max
        if phase == 'test':
            self.dataset_size = self.images_source_size

        if self.phase == 'train':
            transforms_source = [RandomCrop(args.patch_size_down)] # RandomCrop
            transforms_target = [RandomCrop(args.patch_size_down // self.args.scale)]
            if args.flip:
                transforms_source.append(RandomHorizontalFlip())
                transforms_source.append(RandomVerticalFlip())
                transforms_target.append(RandomHorizontalFlip())
                transforms_target.append(RandomVerticalFlip())
        else:
            transforms_source = []
            transforms_target = []

        transforms_source.append(ToTensor())
        transforms_source.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        self.transforms_source = Compose(transforms_source)

        transforms_target.append(ToTensor())
        transforms_target.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        self.transforms_target = Compose(transforms_target)

        if phase == 'train':
            print('Source: %d, Target: %d images' % (self.images_source_size, self.images_target_size))
        else:
            print('Source: %d' % (self.images_source_size))

    def __getitem__(self, index):
        index_source = index % self.images_source_size
        index_target = random.randint(0, self.images_target_size - 1)  ## for randomness

        data_source, fn = self.load_img(self.images_source[index_source])
        data_target, _ = self.load_img(self.images_target[index_target], domain='target')

        return data_source, data_target, fn

    def load_img(self, img_name, input_dim=3, domain='source'):
        ## loading images
        img = self.padding(Image.open(img_name).convert('RGB'))
        fn = img_name.split('/')[-1]
        ## apply different transfomation along domain
        if domain == 'source':
            img = self.transforms_source(img)
        else:
            img = self.transforms_target(img)

        ## rotating
        rot = self.args.rot and random.random() < 0.5
        if rot:
            img = img.transpose(1, 2)

        ## flipping
        flip_h = self.args.flip and random.random() < 0.5
        flip_v = self.args.flip and random.random() < 0.5
        if flip_h:
            img = torch.flip(img, [2])
        if flip_v:
            img = torch.flip(img, [1])

        return img, fn

    def padding(self, img):
        img_arr = np.array(img)
        h, w, c = img_arr.shape
        if h < self.args.patch_size_down:
            h = self.args.patch_size_down
        if w < self.args.patch_size_down:
            w = self.args.patch_size_down
        patch_img = np.zeros((h, w, c))
        patch_img[:img_arr.shape[0], :img_arr.shape[1], :] = img_arr
        patch_img = Image.fromarray(patch_img.astype(np.uint8))
        return patch_img

    def __len__(self):
        if self.phase == 'train':
            return self.dataset_size  # one epoch for two cycle of training dataset
        else:
            return self.dataset_size


class paired_dataset(data.Dataset):  # only for joint SR
    def __init__(self, args):
        self.dataroot = args.test_dataroot
        self.args = args
        self.crop = args.crop

        if args.realsr:
            test_hr = args.test_lr
        else:
            if args.test_hr is None:
                raise NotImplementedError("test_hr set should be given")
            test_hr = args.test_hr

        ## HR
        images_hr = sorted(os.listdir(os.path.join(self.dataroot, test_hr)))
        images_hr = images_hr[int(args.test_range.split('-')[0]) - 1: int(args.test_range.split('-')[1])]
        self.images_hr = [os.path.join(self.dataroot, test_hr, x) for x in images_hr]
        ## LR
        images_lr = sorted(os.listdir(os.path.join(self.dataroot, args.test_lr)))
        images_lr = images_lr[int(args.test_range.split('-')[0]) - 1: int(args.test_range.split('-')[1])]
        self.images_lr = [os.path.join(self.dataroot, args.test_lr, x) for x in images_lr]

        self.images_hr_size = len(self.images_hr)
        self.images_lr_size = len(self.images_lr)

        assert (self.images_hr_size == self.images_lr_size)

        transforms = []
        transforms.append(ToTensor())
        self.transforms = Compose(transforms)

        print('\njoint training option is enabled')
        print('HR set: {},  LR set: {}'.format(args.test_hr, args.test_lr))
        print('number of test images for SR : %d images' % (self.images_hr_size))

    def __getitem__(self, index):
        data_hr, fn_hr = self.load_img(self.images_hr[index])
        data_lr, fn_lr = self.load_img(self.images_lr[index])
        return data_hr, data_lr, fn_lr

    def load_img(self, img_name):
        ## loading images
        img = self.crop_img(Image.open(img_name).convert('RGB'))
        fn = img_name.split('/')[-1]
        ## apply transfomation
        img = self.transforms(img)
        ## rotating and flipping
        rot = self.args.rot and random.random() < 0.5
        flip_h = self.args.flip and random.random() < 0.5
        flip_v = self.args.flip and random.random() < 0.5
        if rot:
            img = img.transpose(1, 2)
        if flip_h:
            img = torch.flip(img, [2])
        if flip_v:
            img = torch.flip(img, [1])
        return img, fn

    def crop_img(self, img):
        img_arr = np.array(img)
        h, w, c = img_arr.shape
        if h <= self.crop and w <= self.crop:
            return img
        if h > self.crop:
            s = int(round((h - self.crop) / 2.))
            img_arr = img_arr[s:s + self.crop, :, :]
        if w > self.crop:
            s = int(round((w - self.crop) / 2.))
            img_arr = img_arr[:, s:s + self.crop, :]
        img = Image.fromarray(img_arr.astype(np.uint8))
        return img

    def __len__(self):
        return self.images_hr_size