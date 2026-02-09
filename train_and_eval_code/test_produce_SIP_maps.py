import torch
import torch.nn.functional as F
import sys
import torch.nn as nn
import numpy as np
import os, argparse
import cv2
from Code.lib.model import SAINet
from Code.utils.data import test_dataset
from Code.utils.options import opt as o
import time
"""

是的，这段代码的主要功能是使用已经训练好的模型（./Checkpoint/SAINet_nju_best.pth）来对指定的数据集进行预测，并将预测结果保存为图像。
具体而言，它对以下数据集进行测试：'NJU2K', 'NLPR', 'SIP', 'STERE', 'DUT-RGBD', 和 'ReDWebTest'，并将结果保存在相应的文件夹中。

"""
parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=256, help='testing size')
parser.add_argument('--gpu_id', type=str, default='0', help='select gpu id')
parser.add_argument('--test_path', type=str, default='/home/accv/tsf/', help='test dataset path') # 到时候六个额外的数据集也放在这个目录下
opt = parser.parse_args(args=[])

dataset_path = opt.test_path

# set device for test
if opt.gpu_id == '0':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print('USE GPU 0')

# load the model
model = SAINet(32, 50)
model.cuda()
#checkpoint='./Checkpoint/SAINet_nju_best.pth'  # 之前训练得到的最好的模型
checkpoint=''
model.load_state_dict(torch.load(checkpoint))
model.eval()

# test
test_datasets = ['SIP']  # 六个额外的数据集

# test_datasets = ['new-imag2.0']

for dataset in test_datasets:
    #save_path = './test_maps_gai/'+dataset+"/"
    save_path = './test_maps/'+dataset+"/"  # 将预测结果保存为图像的路径
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    #image_root = dataset_path + dataset + '/rgb-test/'
    #gt_root = dataset_path + dataset + '/gt-test/'
    #depth_root = dataset_path + dataset + '/d-test/'
    image_root = dataset_path +'/TestDataset/'+ dataset + '/RGB/'  # 六个额外的数据集
    gt_root = dataset_path + '/TestDataset/'+  dataset +'/GT/'
    depth_root = dataset_path +'/TestDataset/'+   dataset + '/depth/'
    test_loader = test_dataset(image_root, gt_root, depth_root, opt.testsize)
    time_sum = 0
    for i in range(test_loader.size):
        image, gt, depth, name, image_for_post = test_loader.load_data()

        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        depth = depth.repeat(1, 3, 1, 1).cuda()
        time_start = time.time()
        pre_res = model(image, depth)
        time_end = time.time()
        time_sum = time_sum+(time_end-time_start)
        res = pre_res[0]
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        if i == test_loader.size-1:
            print('Running time {:.5f}'.format(time_sum/test_loader.size))
            print('Average speed: {:.4f} fps'.format(test_loader.size/time_sum))


        print('save img to: ', save_path + name)
        cv2.imwrite(save_path+name, res * 255)


    print('Test Done!')
