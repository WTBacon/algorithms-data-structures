# coding: utf-8
import requests
import os
import torch
import torchvision
import torchvision.transforms as transforms
from skimage import io, transform,data
from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
from torch.utils.serialization import load_lua
import torch.nn as nn
import torch.optim as optim
import pickle
import torchvision.utils as vutils
import numpy as np
from torchvision import datasets
import torch.nn.functional as F
from scipy import ndimage
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import math
import random
import numbers
from skimage.transform import rescale, resize, downscale_local_mean
import shutil
from sklearn.model_selection import train_test_split
import tifffile
import csv

random.seed(141421356)

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

smp_img_path = '/home/rurararura31/01_satellite/input/'
y_train = pd.read_csv(os.path.join(smp_img_path, 'train_master.tsv'), delimiter='\t')

print(y_train.shape )
print(y_train[y_train.flag==0].shape)
print(y_train[y_train.flag==1].shape)

satellite_train, satellite_valid = train_test_split(y_train, test_size = 0.2, stratify = y_train['flag'])

print('y_train 0 : ', y_train[y_train.flag==0].shape)
print('y_train 1 : ', y_train[y_train.flag==1].shape)
print('satellite_train 0 : ', satellite_train[satellite_train.flag==0].shape)
print('satellite_train 1 : ', satellite_train[satellite_train.flag==1].shape)
print('satellite_valid 0 : ', satellite_valid[satellite_valid.flag==0].shape)
print('satellite_valid 1 : ', satellite_valid[satellite_valid.flag==1].shape)

#satellite_train = satellite_train.head(10000)
#satellite_valid = satellite_valid.head(10000)


# # ここからモジュール化する

train_img_path = os.path.join(smp_img_path, 'train')

#====================== ここから関数 =======================#
IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

def is_image_file(filename):
    """Checks if a file is an image.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)

def make_train_dataset(dir, y_train):
    images = []
    for i, row in y_train.iterrows():
        path = os.path.join(dir, row[0])
        train_y_num = row[1]
        if os.path.isfile(path):
            if train_y_num == 1:
                item = [(path, train_y_num)] * 4
                images.extend(item)
            else:
                item = (path, train_y_num)
                images.append(item)
    return images

def make_valid_dataset(dir, y_train):
    images = []
    for i, row in y_train.iterrows():
        path = os.path.join(dir, row[0])
        train_y_num = row[1]
        if os.path.isfile(path):
            if train_y_num == 1:
                item = [(path, train_y_num)]
                images.extend(item)
            else:
                item = (path, train_y_num)
                images.append(item)
    return images

"""pathをloadするときのdefaultの設定"""
def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return tifffile.imread(path)

#====================== ここからクラス =======================#
class train_SatelliteDataset(Dataset):

    def __init__(self, root, y_train,
                 transform=None, loader=default_loader):

        """imgs : [(画像のpath, 画像のclass), ...] """
        self.imgs = make_train_dataset(root, y_train)

        if len(self.imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        self.root = root
        self.transform = transform
        self.loader = loader

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        path, target = self.imgs[idx]
        img = self.loader(path)

        # numpy image: H x W x C
        # torch image: C X H X W
        #img = img.transpose((2, 0, 1))

        if self.transform:
            img = self.transform(img)

        return img, target

class valid_SatelliteDataset(Dataset):

    def __init__(self, root, y_train,
                 transform=None, loader=default_loader):

        """imgs : [(画像のpath, 画像のclass), ...] """
        self.imgs = make_valid_dataset(root, y_train)

        if len(self.imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        self.root = root
        self.transform = transform
        self.loader = loader

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        path, target = self.imgs[idx]
        img = self.loader(path)

        # numpy image: H x W x C
        # torch image: C X H X W
        #img = img.transpose((2, 0, 1))

        if self.transform:
            img = self.transform(img)

        return img, target

class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image):
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        return image


class RandomSizedCrop(object):
    """Crop the given PIL.Image to random size and aspect ratio.

    A crop of random size of (0.08 to 1.0) of the original size and a random
    aspect ratio of 3/4 to 4/3 of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size: size of the smaller edge
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size): #, interpolation=Image.BILINEAR):
        self.size = size
        # self.interpolation = interpolation

    def __call__(self, img):
        for attempt in range(10):
            area = img.shape[0] * img.shape[1]
            target_area = random.uniform(0.08, 1.0) * area
            aspect_ratio = random.uniform(3. / 4, 4. / 3)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.shape[0] and h <= img.shape[1]:
                x1 = random.randint(0, img.shape[0] - w)
                y1 = random.randint(0, img.shape[1] - h)

                img = img[x1: (x1 + w), y1: (y1 + h)]
                assert(img.shape[:2] == (w, h))

                return resize(img, (self.size, self.size), mode='reflect') #, self.interpolation)

        # Fallback
        scale = Scale(self.size) # , interpolation=self.interpolation)
        crop = CenterCrop(self.size)
        return crop(scale(img))

class CenterCrop(object):
    """Crops the given PIL.Image at the center.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be cropped.

        Returns:
            PIL.Image: Cropped image.
        """
        w, h = img.shape[:2]
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        #return img.crop((x1, y1, x1 + tw, y1 + th))
        return img[x1:(x1 + tw),  y1:(y1 + th)]

class Scale(object):
    """Rescale the input PIL.Image to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (w, h), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size):#, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        #self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be scaled.

        Returns:
            PIL.Image: Rescaled image.
        """
        if isinstance(self.size, int):
            w, h = img.shape[:2]
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return resize(img, (ow, oh), mode='reflect')#, self.interpolation)
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return resize(img, (ow, oh), mode='reflect') # , self.interpolation)
        else:
            return resize(img, (self.size), mode='reflect')

class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL.Image randomly with a probability of 0.5."""

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be flipped.

        Returns:
            PIL.Image: Randomly flipped image.
        """
        rotation = random.random()
        if rotation < 0.25:
            return ndimage.rotate(img, 90, reshape=False,  mode='wrap')
        elif rotation < 0.5:
            return ndimage.rotate(img, 180, reshape=False,  mode='wrap')
        elif rotation < 0.75:
            return ndimage.rotate(img, 270, reshape=False,  mode='wrap')
        return img

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, image):

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return torch.from_numpy(image).float()

# # Transfer Learning tutorial (Pytorch Tutorials)
# License: BSD
# Author: Sasank Chilamkurthy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        RandomCrop(28),
        RandomHorizontalFlip(),
        ToTensor(),
    ]),
    'val': transforms.Compose([
        CenterCrop(28),
        ToTensor(),
    ]),
}

data_img_path = os.path.join(smp_img_path, 'wtb_dataset')
make_dir(data_img_path)
"""
# =================== dataset make ========================#
image_datasets = {}
image_datasets['train'] = train_SatelliteDataset(root=train_img_path, y_train=satellite_train, transform = data_transforms['train'])
image_datasets['val'] = valid_SatelliteDataset(root=train_img_path, y_train=satellite_valid, transform = data_transforms['val'])
dataloaders = {}
dataloaders['train'] = torch.utils.data.DataLoader(image_datasets['train'], batch_size=512, shuffle=True, num_workers = 8)
dataloaders['val'] = torch.utils.data.DataLoader(image_datasets['val'], batch_size=512, num_workers = 8)
#dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=512, shuffle=True, num_workers = 8) for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

print ('train dataset size : ', dataset_sizes['train'])
print ('valid dataset size : ', dataset_sizes['val'])

with open(os.path.join(data_img_path, 'image_datasets_512.pickle'), mode = 'wb') as f:
    pickle.dump(image_datasets, f)
with open(os.path.join(data_img_path, 'dataloaders_512.pickle'), mode = 'wb') as f:
    pickle.dump(dataloaders, f)
print('save dataset!')
# =================== dataset make ========================#
"""

# =================== dataset load ========================#
with open(os.path.join(data_img_path, 'image_datasets_512.pickle'), mode = 'rb') as f:
    image_datasets = pickle.load(f)
with open(os.path.join(data_img_path, 'dataloaders_512.pickle'), mode = 'rb') as f:
    dataloaders = pickle.load(f)

dataset_sizes = { x : len(image_datasets[x]) for x in ['train', 'val']}

print('train dataset size : ', dataset_sizes['train'])
print('valid dataset size : ', dataset_sizes['val'])
print('load dataset!')
# =================== dataset load ========================#


use_gpu = torch.cuda.is_available()
model_img_path = os.path.join(smp_img_path, 'model')
make_dir(model_img_path)

print('You can use GPU : ', use_gpu)

def IOU(preds, epoch_labels):
    TT = np.float(np.sum(preds[np.where(epoch_labels == 1)[0]] == 1))
    FT = np.float(np.sum(preds[np.where(epoch_labels == 0)[0]] == 1))
    TF = np.float(np.sum(preds[np.where(epoch_labels == 1)[0]] == 0))
    print('TT : ', TT)
    print('FT : ', FT)
    print('TF : ', TF)
    if TT == 0.:
        return 0.
    else:
        return TT / (TT+TF+FT)

def train_model(model, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_IOU = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('Learning Rate: {}'.format(optimizer.param_groups[0]['lr']))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_iou = 0.0
            running_preds_list = []
            running_labels_list = []

            TT = 0.0
            FT = 0.0
            TF = 0.0

            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # outputs = model(inputs).squeeze()
                outputs = model(inputs)

                """ torch.max(input, dim, keepdim=False, out=None) -> (Tensor, LongTensor)"""
                """_, preds = torch.max(outputs.data, 1)"""
                """nn.CrossEntropyLoss() includes LogSoftMax."""

                weight_labels = labels.data.cpu().numpy()
                weight_class = torch.FloatTensor([1-sum(weight_labels)/ len(weight_labels), sum(weight_labels)/len(weight_labels)]).cuda()
                #print(weight_class)

                criterion = nn.CrossEntropyLoss(weight=weight_class)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs.data, 1)


                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.cpu().data[0] * inputs.size(0)

                if phase == 'val':
                    running_preds_list += list(preds.cpu().numpy())
                    running_labels_list += list(labels.data.cpu().numpy())

            #import pdb; pdb.set_trace()

            print('dataset phase: ', phase, ' dataset size: ',dataset_sizes[phase])
            if phase == 'train':
                epoch_loss = running_loss / dataset_sizes[phase]
                print('{} Loss: {:.4f}'.format(phase, epoch_loss))

            if phase == 'val':
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_preds = np.array(running_preds_list)
                epoch_labels = np.array(running_labels_list)
                epoch_IOU = IOU(epoch_preds, epoch_labels)

                print('{} Loss: {:.4f} IOU: {:.4f}'.format(phase, epoch_loss, epoch_IOU))
                scheduler.step(epoch_IOU)

            # deep copy the model
            if phase == 'val' and epoch_IOU > best_IOU:
                best_IOU = epoch_IOU
                best_model_wts = copy.deepcopy(model.state_dict())

        torch.save(model.state_dict(), os.path.join(model_img_path, 'epoch{}_state_dict.pth'.format(epoch)))

        with open(os.path.join(model_img_path, 'all_loss_IOU.txt'), 'a+') as f:
            writer = csv.writer(f, lineterminator = '\n')
            writer.writerow(['epoch: {} Loss: {:.4f} IOU: {:.4f} Learning Rate: {}'.format(epoch, epoch_loss, epoch_IOU, optimizer.param_groups[0]['lr'])])

        time_elapsed = time.time() - since
        print('Training time... {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val IOU: {:4f}'.format(best_IOU))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def visualize_model(model, num_images=6):
    images_so_far = 0
    fig = plt.figure()

    for i, data in enumerate(dataloaders['val']):
        inputs, labels = data
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        outputs = model(inputs)
        # _ : value, preds : index
        _, preds = torch.max(outputs.data, 1)

        for j in range(inputs.size()[0]):
            images_so_far += 1
            ax = plt.subplot(num_images//2, 2, images_so_far)
            ax.axis('off')
            ax.set_title('predicted: {}'.format(class_names[preds[j]]))
            imshow(inputs.cpu().data[j])

            if images_so_far == num_images:
                return

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 256

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 128, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    # resnet18の場合: [2,2,2,2]
    # stride数を変えたblockをnum_blocks分生成する
    def _make_layer(self, block, planes, num_blocks, stride):
        # block : ブロックの種類
        # planes : resnetの数
        # num_blocks : 各レイヤーのブロックの数
        # stride : stride数
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# []
def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2]) # []：各layerのblockの数

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])



# ### ============= model =============

model_ft = ResNet50()
model_ft.conv1 = nn.Conv2d(7, 256, kernel_size=3, stride=1, padding=1, bias=False)
model_ft.bn1 = nn.BatchNorm2d(256)

# model_ft.linear  = nn.Linear(512, 2) # resnet18
model_ft.linear  = nn.Linear(2048, 2) # resnet50
print(model_ft)

if use_gpu:
    model_ft = nn.DataParallel(model_ft)
    model_ft = model_ft.cuda()

#criterion = nn.CrossEntropyLoss()

#param = torch.load('/home/rurararura31/01_satellite/input/model_part1/epoch44_state_dict.pth')
#model_ft.load_state_dict(param)

# Observe that all parameters are being optimized
#optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.1)
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.1, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
#exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=3000, gamma=0.1)
exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='max', verbose=True, patience=20, factor=0.5)
# optimizer_ft, mode='max', verbose=True, patience=10, factor=0.8, cooldown=3

model_ft = train_model(model_ft, optimizer_ft, exp_lr_scheduler, num_epochs=500)

with open(os.path.join(model_img_path, 'resnet_bestModel.pickle'), mode = 'wb') as f:
    pickle.dump(model_ft, f)
