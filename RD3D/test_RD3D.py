# RDD3D测试
import copy
import os

import torch.nn.functional as F
import torch
import torchvision 
from RD3D.rd3d import RD3D
from  Code.utils.options import opt
from RD3D.data import get_loader

if __name__ == '__main__':
# RD3D的测试
    RD3D_resnet = torchvision.models.resnet50(pretrained=True)
    RD3D_model = RD3D(32, copy.deepcopy(RD3D_resnet))
    
    optimizer = torch.optim.Adam(RD3D_model.parameters(), opt.lr)
    
    RD3D_image_root = os.path.join(opt.rgb_label_root)
    RD3D_gt_root = os.path.join(opt.gt_label_root)
    RD3D_depth_root = os.path.join(opt.depth_label_root)
    CE = torch.nn.BCEWithLogitsLoss().cuda()
    
    RD3D_train_loader = get_loader(RD3D_image_root, RD3D_gt_root, RD3D_depth_root, batchsize=opt.batchsize,
                                trainsize=opt.trainsize)
    print("rgb path {}, gt path {}, depth path {}".format(RD3D_image_root, RD3D_gt_root, RD3D_depth_root))
    
    
    size_rates = [0.75, 1, 1.25]
    RD3D_model.train()
    for i, pack in enumerate(RD3D_train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            images, gts, depths = pack
            images = images.cuda()
            gts = gts.cuda()
            depths = depths.cuda()

            # multi-scale training samples
            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                images = images.unsqueeze(2)  # 原始张量的第 2 维插入一个大小为 1 的新维度
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)

                depths = F.upsample(depths, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                depths = depths.unsqueeze(2)
                images = torch.cat([images, depths], 2)

            if rate == 1:
                images = images.unsqueeze(2)  # （batchsize，3，1，352，352）
                depths = depths.unsqueeze(2)  # （batchsize，3，1，352，352）
                images = torch.cat([images, depths], 2) # (batch_size, 3, 2, height, width)

            # forward
            pred_s = RD3D_model(images)
            loss = CE(pred_s, gts)

            loss.backward()
            optimizer.step()
