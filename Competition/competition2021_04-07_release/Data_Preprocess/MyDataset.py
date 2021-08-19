'''
module :
        Padding
        Interpolate
'''
import os, sys, math

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision

py_version = float((sys.version)[:3])

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, root, img_info_tab = 'basic_image_tab.csv', transforms=None, 
                    train = True, scale = 0.8, IsAug = None, shape = (64,64), test_data = None, method = None):
        super().__init__()
        '''

            test_data:
                if test_data not None, the argument is Testing Dataset Folder ,ex: test_stage_data_0615
        '''
        data_frame = pd.read_csv(root + '/' + img_info_tab, encoding = 'big5')
        imgs = data_frame['img_name'].values.tolist()
        labels = data_frame['img_token'].values.tolist()
        self.imgs = imgs
        self.labels = labels
        
        self.root = os.path.join(root, 'ImageDataset', 'training_set')
        self.transforms = transforms
        if test_data != None:
            self.root = os.path.join(root, 'ImageDataset', 'testing_set', test_data)
        self.train = train
        self.count = int(len(self.imgs)*scale) # train_count
        self.IsAug = IsAug
        self.height, self.width = shape # 
        self.method = method 

    def __len__(self):
        # return number of dataset 
        if self.train == True:
            return self.count
        else:
            return len(self.imgs) - self.count # valid_count = total - train_count

    def __getitem__(self, index):
        
        # get img && label
        if self.train == True:
            img_path = os.path.join(self.root , self.imgs[index])
            img = Image.open(img_path)
            label = self.labels[index]
        else:
            img_path = os.path.join(self.root , self.imgs[self.count+index])
            img = Image.open(img_path)
            label = self.labels[self.count+index]
        #
        transform_toTensor = torchvision.transforms.ToTensor()
        transforms_toImage = torchvision.transforms.ToPILImage()
        if self.IsAug != None:
            # [1]augument on training dataset
            if self.train == True:
                if self.transforms != None:
                    img = self.transforms(img) # 使用隨機剪裁、noise、隨機翻轉...
                else:
                    transform_toResize = torchvision.transforms.Resize((self.height, self.width))
                    img = transform_toResize(img)
                    # 下列這些條件是原先訓練資料有旋轉90度的，不過已不使用，做為稀疏資料不影響，且無明顯提升
                    # ### 無效原因，可能與Resize有差異, 訓練以及測試、驗證的input 為padding 或Interpolate縮放的資料 ###
                    if self.IsAug == 'Rotate90':
                        img = img.transpose(Image.ROTATE_90)
                    elif self.IsAug == 'Rotate180':
                        img = img.transpose(Image.ROTATE_180)
                    elif self.IsAug == 'Rotate270':
                        img = img.transpose(Image.ROTATE_270)
                    elif self.IsAug == 'Rotate_pos_20':
                        img = img.rotate(-20) #順時
                    elif self.IsAug == 'Rotate_neg_20':
                        img = img.rotate(20)
                    else:
                        raise ValueError('>> Error Data Augument')
                    img = transform_toTensor(img)
                return img, label
            else:
                raise ValueError('>>Get Dataset Error : If used Augumentation data, please check \'train=True\'. ')
        else:

            if self.transforms != None:
                img = self.transforms(img)
            else:
                
                gt_height , gt_width = np.array(img).shape[0], np.array(img).shape[1] # ground truth
                if self.method == None:
                    # [2]image on center ,padding round img
                    if self.width > gt_width:
                        add_h = self.height - gt_height
                        add_w = self.width - gt_width 
                        if add_h % 2 == 0:
                            add_top = add_h//2
                            add_bottom = add_h//2
                        else:
                            add_top = add_h//2
                            add_bottom = add_h//2 + 1

                        if add_w % 2 == 0:
                            add_left = add_w//2
                            add_right = add_w//2
                        else:
                            add_left = add_w//2
                            add_right = add_w//2+1
                        padding = (add_left, add_top, add_right, add_bottom)
                        transforms_pad = torchvision.transforms.Pad(padding, fill = 0, padding_mode="constant")
                        img = transforms_pad(img)
                    else:
                        transform_toResize = torchvision.transforms.Resize((self.height, self.width))
                        img = transform_toResize(img)
                    img = transform_toTensor(img)

                elif self.method == 'Interpolate':
                    # [3]image on center and using Interpolate to scale img height / width ; pad 0 to square
                    scale_f = self.height/gt_height
                    # 長固定67，故使用64X64目標的話 縮放比會小於1，偏偏剛好有數據寬度為1，此時使用插值會報錯
                    if self.width > gt_width and gt_width * scale_f >= 1:
                        # image->tensor , get tensor to use torch func. 
                        img = transform_toTensor(img)
                        if py_version >= 3.7:
                            img = nn.functional.interpolate(img.unsqueeze(0), scale_factor = scale_f, mode = 'bilinear', align_corners=True ,recompute_scale_factor=True)
                        else:
                            img = nn.functional.interpolate(img.unsqueeze(0), scale_factor = scale_f, mode = 'bilinear', align_corners=True)
                        pad_width = self.width - img.size()[3] # [bs, c, h, w]
                        if pad_width % 2 == 0:
                            padding = (pad_width //2, 0, pad_width//2, 0) # left, top, right, botton
                        else:
                            padding = (pad_width //2, 0, pad_width//2+1, 0)
                        data_transforms = torchvision.transforms.Compose([
                            torchvision.transforms.Pad(padding, fill = 0, padding_mode='constant'),
                            torchvision.transforms.ToTensor()
                        ])
                        scale_img = transforms_toImage(img.squeeze(0))
                        img = data_transforms(scale_img)
                        
                    else:  
                        img = transform_toTensor(img)
                        if py_version >= 3.7:
                            img = nn.functional.interpolate(img.unsqueeze(0), scale_factor = scale_f, mode = 'bilinear', align_corners=True ,recompute_scale_factor=True)
                        else:
                            img = nn.functional.interpolate(img.unsqueeze(0), (self.height, self.width), mode = 'bilinear', align_corners=True)
                        img = img.squeeze(0)
            # padding or interpolate  
            return img, label

if __name__ == '__main__':
    print(py_version)
    # # sample
    # root = os.getcwd()
    # Aug_data = MyDataset(root, IsAug = 'Rotate90')
    # data_loader = torch.utils.data.DataLoader(dataset = Aug_data ,batch_size = 4,shuffle = False) 
    # for i, (data, label) in enumerate(data_loader):
        
    #     plt.figure(figsize =(8,4))
    #     index = 0
    #     for sample in data:
                
    #         plt.subplot(1, 4, index+1)
    #         plt.imshow(torchvision.transforms.ToPILImage()(sample))
    #         index += 1
    #     if i == 1:
    #         break
    # # =====================        
    # train_data = MyDataset(root, shape = (96, 96), method = None)
    # data_loader = torch.utils.data.DataLoader(dataset = train_data ,batch_size = 4,shuffle = False) 
    # for i, (data, label) in enumerate(data_loader):
        
    #     plt.figure(figsize =(8,4))
    #     index = 0
    #     for sample in data:
                
    #         plt.subplot(1, 4, index+1)
    #         plt.imshow(torchvision.transforms.ToPILImage()(sample))
    #         index += 1
    #     if i == 1:
    #         break
    # # =====================
    # train_data = MyDataset(root, shape = (96, 96), method = 'Interpolate')
    # data_loader = torch.utils.data.DataLoader(dataset = train_data ,batch_size = 4,shuffle = False) 
    # for i, (data, label) in enumerate(data_loader):
        
    #     plt.figure(figsize =(8,4))
    #     index = 0
    #     for sample in data:
                
    #         plt.subplot(1, 4, index+1)
    #         plt.imshow(torchvision.transforms.ToPILImage()(sample))
    #         index += 1
    #     if i == 1:
    #         break    