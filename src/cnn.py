from __future__ import print_function, division
import os
import torch
import torch.nn as nn
import pandas as pd
from skimage import io, transform, filters
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pickle
import VizGraph as viz
class SegDataset(Dataset):
    def __init__(self, image_folder, target_folder, transform=None):
        '''
        Args:
            image_folder (string): path to input images
            labels (dict): {img1: {...}, img2: {...}, ...}
            transform (callable, optional): optional transform to be
                                    applied on a sample
        '''
        self.image_folder = image_folder
        self.target = pickle.load( open(os.path.join(target_folder, "target.p"), "rb" ) )
        self.transform = transform
    
    def __len__(self):
        return len(self.target)
    
    def __getitem__(self, image_name):
        if image_name in os.listdir(self.image_folder):
            image_path = os.path.join(self.image_folder, image_name)
            image = io.imread(image_path, as_grey=True)
        else:
            print(image_name,'does not exist!')
        label = self.target['labels'][image_name]
        gt_aff = self.target['gt_affs'][image_name]
        sample = {'image': image, 'label': label, 'gt_aff': gt_aff}
        if self.transform:
            sample = self.transform(sample)
        return sample

class MapData(object):
    '''
    Transform image to tensors of features and labels to tensors of binary edges map
    '''
    def __call__(self, sample):
        image, label, gt_aff = sample['image'], sample['label'], sample['gt_aff']
        
        #make new features here, for now only try edge map from sobel
        edge_map = filters.sobel(image)

        #concatenate map together as new image
        image = np.dstack((image, edge_map))

        #swap axis
        #numpy image: height x width x channel
        #torch image: chanel x height x width
        image = image.transpose((2, 0, 1))
        gt_aff = gt_aff.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image), 'label': label, 'gt_aff': torch.from_numpy(gt_aff)}



        






#Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#Hyper parameters
num_epouchs = 5
batch_size = 10
learning_rate = 0.001

class ConvNet(nn.Module):
    def __init__(self, num_edges = 2*(100**2 - 100)):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=2, out_channels=1, kernel_size=(2,1,1))
        self.conv2H = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1,2), stride=1, padding=0)
        self.conv2V = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(2,1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1(x)
        out = torch.cat((self.conv2H(out), torch.transpose(self.conv2V(out),2,3)), dim=1)
        




if __name__ == '__main__':
    print('Init')
    
    print('Making dataset...')
    segdata = SegDataset(image_folder='./synimage/original', target_folder='./synimage/groundtruth', transform=MapData())
    print('Done')

    

    ind = segdata['s1.bmp']['image']
    ind = torch.unsqueeze(ind,dim=0)
    ind = ind.float()
    model = torch.nn.Sequential(
        nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1),
        nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(2,1))
        )
    out = model(ind)

    print(out.detach().numpy().shape)

    print("Exit")