"""
Code: https://github.com/mczhuge/ICON
Author: mczhuge
Desc: Core code for validation
"""
import torch.nn.functional as F
import os
import sys
import cv2
from tqdm import tqdm
import metrics as M
import json
import argparse

def main():
    # args = parser.parse_args()

    FM = M.Fmeasure_and_FNR()
    WFM = M.WeightedFmeasure()
    SM = M.Smeasure()
    EM = M.Emeasure()
    MAE = M.MAE()
    #FNR = M.FNR()

    # method = args.method
    # dataset = args.dataset
    #attr = args.attr

    #SOD

    gt_root = os.path.join("/home/accv/tsf/TestDataset/") # 六个额外的数据集的路径
    # gt_root = os.path.join("/home/accv/tsf") # 数据集的路径

    # gt_root = os.path.join(r"F:\BaiduNetdiskDownload\DCF\RGB-D SOD test_data\test_data/")
    pred_root = os.path.join("./test_maps/")  # 将预测结果保存为图像的文件路径
    #test_datasets = ["NJU2K","DES","LFSD", "NLPR", "ReDwebTest","SSD", "SIP", "STERE"]
    # test_datasets = ["DES","DUT-RGBD", "LFSD", "NLPR","SSD", "SIP", "STERE"]
    # test_datasets = [ "DES","NLPR","DUT-RGBD", "LFSD", "SIP","SSD","STERE"]
    # test_datasets = ["DES","NJU2K","NLPR","SSD", "SIP","STERE"]
    # test_datasets = ["SSD"]
    # test_datasets = ['NJU2K', "DES", "DUT-RGBD", "LFSD", "NLPR","SSD","STERE"]
    # test_datasets = ['NJUD', "RGBD135", "DUT", "LFSD", "NLPR","SSD", "SIP", "STERE1000"]
    # test_datasets = ["DUT-RGBD","ReDwebTest"]
    # test_datasets = ["DUT-RGBD","ReDwebTest","SSD"]
    # test_datasets = ["DUT-RGBD",'ReDwebTest']
    # test_datasets = ["NJU2K"]
    test_datasets = ["STERE"]
    #test_datasets = ["new-imag2.0"]
    print(gt_root)
    print(pred_root)

    results=[]
    for dataset in test_datasets:

        pred_dataset_root=pred_root+"/"+dataset+"/"
        # pred_dataset_root = pred_root +"/"
        gt_dataset_root=gt_root+"/"+dataset+"/GT/"
        #gt_dataset_root=gt_root+"/"+dataset+"/gt-test/" # 数据ground truth的路径
        gt_name_list = sorted(os.listdir(gt_dataset_root))
        for gt_name in tqdm(gt_name_list, total=len(gt_name_list)): # gt_name_list 是包含所有真实标签图像文件名的列表。
            #print(gt_name)  d 文件夹是tiff，gt文件夹是png，rgb文件夹是bmp
            gt_path = os.path.join(gt_dataset_root, (gt_name)) # gt_path 和 pred_path 生成真实标签图像和预测结果图像的路径。
            pred_path = os.path.join(pred_dataset_root, (gt_name[:-4]+".png")) 
            gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE) # 使用 cv2.imread 读取这两幅图像，cv2.IMREAD_GRAYSCALE 表示以灰度模式读取图像。
            gt_width, gt_height = gt.shape
            pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
            pred_width, pred_height = pred.shape
            #print(gt.shape, pred.shape)C:/Users/me/Desktop/model/MobileSal/SalMap/NJU2K/000001_left.png
            if gt.shape != pred.shape:
                cv2.imwrite( os.path.join(pred_dataset_root, gt_name), cv2.resize(pred, gt.shape[::-1]))
                pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

            FM.step(pred=pred, gt=gt) # 每个函数都会比较预测结果 pred 和真实标签 gt，并更新相应的统计信息。
            WFM.step(pred=pred, gt=gt)
            SM.step(pred=pred, gt=gt)
            EM.step(pred=pred, gt=gt)
            MAE.step(pred=pred, gt=gt)
            #FNR.step(pred=pred, gt=gt)

        fm = FM.get_results()[0]['fm']
        wfm = WFM.get_results()['wfm']
        sm = SM.get_results()['sm']
        em = EM.get_results()['em']
        mae = MAE.get_results()['mae']
        fnr = FM.get_results()[1]

        # Method_r = str(args.method)
        Dataset_r = str(dataset)
        Smeasure_r = str(sm.round(4))
        Wmeasure_r = str(wfm.round(4))
        MAE_r = str(mae.round(4))
        adpEm_r = str(em['adp'].round(4))
        meanEm_r = str('-' if em['curve'] is None else em['curve'].mean().round(4))
        maxEm_r = str('-' if em['curve'] is None else em['curve'].max().round(4))
        adpFm_r = str(fm['adp'].round(4))
        meanFm_r = str(fm['curve'].mean().round(4))
        maxFm_r = str(fm['curve'].max().round(4))
        fnr_r = str(fnr.round(4))


        eval_record = str(
            # 'Method:'+ Method_r + ','+
            'Dataset:'+ Dataset_r + '||'+


            'MAE:'+ MAE_r + '; '+
            'maxFm:' + maxFm_r+ '; '+
            'maxEm:' + maxEm_r + '; ' +
            'Smeasure:' + Smeasure_r + '; ' +
            'wFmeasure:' + Wmeasure_r + '; ' +
            'meanEm:' + meanEm_r + '; ' +
            'fnr:'+ fnr_r + '||' +
            'adpFm:' + adpFm_r + '; ' +
            'adpEm:'+ adpEm_r + '; '+
            'meanEm:'+ meanEm_r + '; '+
            'adpFm:'+ adpFm_r + '; '+
            'meanFm:'+ meanFm_r + '; '
             )


        print(eval_record)
        print('#'*50)
        results.append(eval_record)
        txt =pred_dataset_root+"eval.txt" # 保存预测结果图像的文件路径下
        f = open(txt, 'a')
        f.write(eval_record)
        f.write("\n")
        f.close()

    txt = pred_root + "/results.txt"  # 整个test_maps文件夹下保存的所有结果的文件
    f = open(txt, 'a')
    f.write('#' * 200)
    f.write("\n")
    for i in range(len(results)):
        f.write(results[i])
        f.write("\n")
    f.close()
if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--method", default='SAIN')
    # parser.add_argument("--dataset", default='Train')
    #parser.add_argument("--attr", default='SOC-AC')
    main()
