import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import ImageEnhance

#several data augumentation strategies
"""
这些函数主要用于对输入图像进行数据增强处理，帮助提升模型的泛化能力。常见的增强方法包括：

    随机翻转 (cv_random_flip)；
    随机裁剪 (randomCrop)；
    随机旋转 (randomRotation)；
    色彩增强 (colorEnhance)；
    添加高斯噪声 (randomGaussian)；
    添加椒盐噪声 (randomPeper)。
这些操作通常用于训练阶段，帮助训练模型时能够从不同的角度和变换下看到多样化的数据，从而提升模型的鲁棒性。
"""
def cv_random_flip(img, label,depth):
    flip_flag = random.randint(0, 1)
    # flip_flag2= random.randint(0,1)
    #left right flip
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT) # transpose函数交换行、列
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
        depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
    #top bottom flip
    # if flip_flag2==1:
    #     img = img.transpose(Image.FLIP_TOP_BOTTOM)
    #     label = label.transpose(Image.FLIP_TOP_BOTTOM)
    #     depth = depth.transpose(Image.FLIP_TOP_BOTTOM)
    return img, label, depth
def randomCrop(image, label,depth):
    border=20
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width-border , image_width)
    crop_win_height = np.random.randint(image_height-border , image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), label.crop(random_region),depth.crop(random_region)
def randomRotation(image,label,depth):
    mode=Image.BICUBIC
    if random.random()>0.8:
        random_angle = np.random.randint(-15, 15)
        image=image.rotate(random_angle, mode)
        label=label.rotate(random_angle, mode)
        depth=depth.rotate(random_angle, mode)
    return image,label,depth
def colorEnhance(image):
    bright_intensity=random.randint(5,15)/10.0
    image=ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity=random.randint(5,15)/10.0
    image=ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity=random.randint(0,20)/10.0
    image=ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity=random.randint(0,30)/10.0
    image=ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image
def randomGaussian(image, mean=0.1, sigma=0.35):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im
    img = np.asarray(image)
    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])
    return Image.fromarray(np.uint8(img))
def randomPeper(img):

    img=np.array(img)
    noiseNum=int(0.0015*img.shape[0]*img.shape[1])
    for i in range(noiseNum):

        randX=random.randint(0,img.shape[0]-1)  

        randY=random.randint(0,img.shape[1]-1)  

        if random.randint(0,1)==0:  

            img[randX,randY]=0  

        else:  

            img[randX,randY]=255 
    return Image.fromarray(img)  # 返回数据类型是Image


# dataset for training
"""
这段注释表明当前的数据加载器（loader）没有使用归一化后的深度图进行训练和测试。如果使用归一化后的深度图，模型的性能可能会有所提升
"""
#The current loader is not using the normalized depth maps for training and test. If you use the normalized depth maps
#(e.g., 0 represents background and 1 represents foreground.), the performance will be further improved.

class SalObjDataset(data.Dataset):
    def __init__(self, image_root, gt_root,depth_root, trainsize):
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')  or f.endswith('.bmp')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.depths=[depth_root + f for f in os.listdir(depth_root) if f.endswith('.bmp')
                    or f.endswith('.png')  or f.endswith('.tiff')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.depths=sorted(self.depths)
        self.filter_files()
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.depths_transform = transforms.Compose([transforms.Resize((self.trainsize, self.trainsize)),transforms.ToTensor()])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        depth=self.binary_loader(self.depths[index])
        image,gt,depth =cv_random_flip(image,gt,depth)
        image,gt,depth=randomCrop(image, gt,depth)
        image,gt,depth=randomRotation(image, gt,depth)
        image=colorEnhance(image)
        # gt=randomGaussian(gt)
        gt=randomPeper(gt) # 
        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        depth=self.depths_transform(depth)
        
        return image, gt, depth # 返回的都是张量

    def filter_files(self):
        """
        对数据集中不符合要求的文件进行过滤，要求rgb，depth，gt三者的尺寸相同
        """
        assert len(self.images) == len(self.gts) and len(self.gts)==len(self.images)
        images = []
        gts = []
        depths=[]
        for img_path, gt_path,depth_path in zip(self.images, self.gts, self.depths):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            depth= Image.open(depth_path)
            if img.size == gt.size and gt.size==depth.size:
                images.append(img_path)
                gts.append(gt_path)
                depths.append(depth_path)
        self.images = images
        self.gts = gts
        self.depths=depths

    def rgb_loader(self, path):
        # 以二进制模式 ('rb') 打开指定路径 path 的图像文件 特别是图像文件，以确保不会对文件内容进行修改
        with open(path, 'rb') as f:
            # 使用 PIL 库 打开图像文件 此时，图像可以是任意格式（如 PNG、JPEG 等），但它会自动识别并加载图像。
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def resize(self, img, gt, depth):
        """
        如果图像的尺寸小于指定的训练大小 self.trainsize，则将图像、标签和深度图都按比例放大到至少 self.trainsize 的大小。
        如果图像的尺寸已经大于或等于 self.trainsize，则保持原尺寸不变。
        """
        assert img.size == gt.size and gt.size==depth.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            # 图像使用双线性插值（Image.BILINEAR）进行缩放，标签和深度图使用最近邻插值（Image.NEAREST）
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST),depth.resize((w, h), Image.NEAREST)
        else:
            return img, gt, depth

    def __len__(self):
        return self.size


###############################################################################
# 0919
#

class SalObjDataset_var(data.Dataset):
    """
    SalObjDataset 类 只生成并返回一个图像、标签和深度图的增强版本。
    SalObjDataset_var 类 会生成并返回两个不同版本的图像、标签和深度图。它应用了多次数据增强，并将增强后的两个版本都返回给调用者。
    """
    def __init__(self, image_root, gt_root,depth_root, trainsize):
        
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.bmp')]
        self.gts    = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.depths = [depth_root + f for f in os.listdir(depth_root) if f.endswith('.bmp') or f.endswith('.png') or f.endswith('.tiff')]
        self.images = sorted(self.images)
        self.gts    = sorted(self.gts)
        self.depths = sorted(self.depths)
        self.filter_files()
        self.size   = len(self.images)
        
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.depths_transform = transforms.Compose([transforms.Resize((self.trainsize, self.trainsize)),transforms.ToTensor()])

    def __getitem__(self, index):
        
        ## read imag, gt, depth
        image0 = self.rgb_loader(self.images[index])
        gt0    = self.binary_loader(self.gts[index])
        depth0 = self.binary_loader(self.depths[index])
        
        
        ##################################################
        ## out1
        ##################################################
        image,gt,depth = cv_random_flip(image0,gt0,depth0)
        image,gt,depth = randomCrop(image, gt,depth)
        image,gt,depth = randomRotation(image, gt,depth)
        image          = colorEnhance(image)
        gt             = randomPeper(gt)
        image          = self.img_transform(image)
        gt             = self.gt_transform(gt)
        depth          = self.depths_transform(depth)

        ##################################################
        ## out1
        ##################################################
        image2,gt2,depth2 = cv_random_flip(image0,gt0,depth0)
        image2,gt2,depth2 = randomCrop(image2, gt2,depth2)
        image2,gt2,depth2 = randomRotation(image2, gt2,depth2)
        image2          = colorEnhance(image2)
        gt2             = randomPeper(gt2)
        image2          = self.img_transform(image2)
        gt2             = self.gt_transform(gt2)
        depth2          = self.depths_transform(depth2)

        
        return image, gt, depth, image2, gt2, depth2

    def filter_files(self):

        
        assert len(self.images) == len(self.gts) and len(self.gts)==len(self.images)
        images = []
        gts = []
        depths=[]
        for img_path, gt_path,depth_path in zip(self.images, self.gts, self.depths):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            depth= Image.open(depth_path)
            if img.size == gt.size and gt.size==depth.size:
                images.append(img_path)
                gts.append(gt_path)
                depths.append(depth_path)
        self.images = images
        self.gts = gts
        self.depths=depths

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def resize(self, img, gt, depth):
        assert img.size == gt.size and gt.size==depth.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST),depth.resize((w, h), Image.NEAREST)
        else:
            return img, gt, depth

    def __len__(self):
        return self.size



class SalObjDataset_var_unlabel(data.Dataset):
    """
    SalObjDataset_var 更倾向于用于标准的有标签监督学习任务，使用 .jpg 和 .bmp 格式的图像文件。
    SalObjDataset_var_unlabel 适用于无标签学习或半监督学习任务，处理 .png 和 .bmp 格式的图像文件，尽管它依然返回了标签图像 (gt)，
    但可能标签的作用不同，更多地可能是作为辅助信息。
    """
    def __init__(self, image_root, gt_root,depth_root, trainsize):
        
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.png') or f.endswith('.bmp')]
        self.gts    = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.depths = [depth_root + f for f in os.listdir(depth_root) if f.endswith('.bmp') or f.endswith('.png')or f.endswith('.tiff')]
        self.images = sorted(self.images)
        self.gts    = sorted(self.gts)
        self.depths = sorted(self.depths)
        self.filter_files()
        self.size   = len(self.images)
        
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.depths_transform = transforms.Compose([transforms.Resize((self.trainsize, self.trainsize)),transforms.ToTensor()])

    def __getitem__(self, index):
        
        ## read imag, gt, depth
        image0 = self.rgb_loader(self.images[index])
        gt0    = self.binary_loader(self.gts[index])
        depth0 = self.binary_loader(self.depths[index])
        
        
        ##################################################
        ## out1
        ##################################################
        image,gt,depth = cv_random_flip(image0,gt0,depth0)
        image,gt,depth = randomCrop(image, gt,depth)
        image,gt,depth = randomRotation(image, gt,depth)
        image          = colorEnhance(image)
        gt             = randomPeper(gt)
        image          = self.img_transform(image)
        gt             = self.gt_transform(gt)
        depth          = self.depths_transform(depth)

        ##################################################
        ## out1
        ##################################################
        image2,gt2,depth2 = cv_random_flip(image0,gt0,depth0)
        image2,gt2,depth2 = randomCrop(image2, gt2,depth2)
        image2,gt2,depth2 = randomRotation(image2, gt2,depth2)
        image2          = colorEnhance(image2)
        gt2             = randomPeper(gt2)
        image2          = self.img_transform(image2)
        gt2             = self.gt_transform(gt2)
        depth2          = self.depths_transform(depth2)

        
        return image, gt, depth, image2, gt2, depth2

    def filter_files(self):

        assert len(self.images) == len(self.gts) and len(self.gts)==len(self.images)
        images = []
        gts = []
        depths=[]
        for img_path, gt_path,depth_path in zip(self.images, self.gts, self.depths):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            depth= Image.open(depth_path)
            if img.size == gt.size and gt.size==depth.size:
                images.append(img_path)
                gts.append(gt_path)
                depths.append(depth_path)
        self.images = images
        self.gts = gts
        self.depths=depths

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def resize(self, img, gt, depth):
        assert img.size == gt.size and gt.size==depth.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST),depth.resize((w, h), Image.NEAREST)
        else:
            return img, gt, depth

    def __len__(self):
        return self.size

#dataloader for training
def get_loader(image_root, gt_root,depth_root, batchsize, trainsize, shuffle=True, num_workers=12, pin_memory=False):

    dataset = SalObjDataset(image_root, gt_root, depth_root,trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize, # 每个训练批次中的样本数
                                  shuffle=shuffle, # 如果为 True，数据会在每个 epoch 之前随机打乱顺序
                                  num_workers=num_workers,
                                  pin_memory=pin_memory) # 如果为 True，将数据加载到固定内存
    return data_loader


#dataloader for training2
## 09-19-2020
def get_loader_var(image_root, gt_root,depth_root, batchsize, trainsize, shuffle=True, num_workers=0, pin_memory=False):

    dataset = SalObjDataset_var(image_root, gt_root, depth_root,trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


def get_loader_var_unlabel(image_root, gt_root,depth_root, batchsize, trainsize, shuffle=True, num_workers=0, pin_memory=False):

    dataset = SalObjDataset_var_unlabel(image_root, gt_root, depth_root,trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


#test dataset and loader
class test_dataset:
    def __init__(self, image_root, gt_root,depth_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.bmp')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                       or f.endswith('.png')]
        self.depths=[depth_root + f for f in os.listdir(depth_root) if f.endswith('.bmp')
                    or f.endswith('.png') or f.endswith('.tiff')]
        self.images = sorted(self.images)  # 字典序排序
        self.gts = sorted(self.gts)
        self.depths=sorted(self.depths)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        # self.gt_transform = transforms.Compose([
        #     transforms.Resize((self.trainsize, self.trainsize)),
        #     transforms.ToTensor()])
        self.depths_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor()])
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)  # .unsqueeze(0) 用于在张量的第一个维度（批次维度）添加一个新的维度。
        gt = self.binary_loader(self.gts[self.index])
        depth=self.binary_loader(self.depths[self.index])
        depth=self.depths_transform(depth).unsqueeze(0)
        name = self.gts[self.index].split('/')[-1]
        image_for_post=self.rgb_loader(self.images[self.index])
        image_for_post=image_for_post.resize(gt.size)
        if name.endswith('.bmp'):
            name = name.split('.bmp')[0] + '.png'
        self.index += 1
        self.index = self.index % self.size
        return image, gt,depth, name,np.array(image_for_post)
# 返回四个元素：
# image：转换后的图像（Tensor），形状为 (1, C, H, W)，即增加了一个维度表示批次大小。
# gt：转换后的标签图像（Tensor），形状为 (1, H, W)，因为标签图像是灰度图。
# depth：转换后的深度图（Tensor），形状为 (1, H, W)。
# name：图像的名称（字符串），用于后续处理或保存。
# image_for_post：调整为与标签图像相同大小的原图（PIL 图像），通常用于后处理阶段。

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
    def __len__(self):
        return self.size

