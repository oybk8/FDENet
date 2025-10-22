"""
Based on https://github.com/asanakoy/kaggle_carvana_segmentation
"""
import torch
import torch.utils.data as data

import cv2
import numpy as np
import os
import random
import setting

random.seed(setting.setting['seed'])

def add_gaussian_noise(image, mean=10, sigma=0.1, u=0.5):
    """
    向图像添加高斯噪声
    :param image: 输入图像
    :param mean: 噪声均值
    :param sigma: 噪声标准差
    :return: 带噪声的图像
    """
    if np.random.random() < u:
        gaussian = np.random.normal(mean, sigma, image.shape).reshape(image.shape[0], image.shape[1], 3)
        noisy = image + gaussian
        noisy = np.clip(noisy, 0, 255)  # 限制值在0-255之间
        return noisy.astype(np.uint8)
    return image

def apply_gaussian_blur(image, kernel_size=5, u=0.5):
    """
    对图像应用高斯模糊
    :param image: 输入图像
    :param kernel_size: 高斯核大小
    :return: 高斯模糊后的图像
    """
    if np.random.random() < u:
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return image

def add_salt_and_pepper_noise(image, salt_vs_pepper=0.05, amount=0.05, u=0.5):
    """
    向图像添加椒盐噪声
    :param image: 输入图像
    :param salt_vs_pepper: 椒盐比例
    :param amount: 噪声比例
    :return: 带椒盐噪声的图像
    """
    if np.random.random() < u:
        noisy = np.copy(image)
        num_salt = np.ceil(amount * image.size * salt_vs_pepper)
        num_pepper = np.ceil(amount * image.size * (1.0 - salt_vs_pepper))

        # 添加盐噪声
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        noisy[coords[0], coords[1], :] = 1

        # 添加椒噪声
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        noisy[coords[0], coords[1], :] = 0

        return noisy.astype(np.uint8)
    return image

def random_crop(image, crop_shape, mask=None):
    image_shape = image.shape
    ret = []
    if crop_shape[0]<image_shape[0]: # 当测试设定不裁剪时，略去裁剪步骤
        nh = np.random.randint(0, image_shape[0] - crop_shape[0])
        nw = np.random.randint(0, image_shape[1] - crop_shape[1])
        image = image[nh:nh + crop_shape[0], nw:nw + crop_shape[1]]
    ret.append(image)
    if mask is not None:
        if crop_shape[0] < image_shape[0]:
            mask = mask[nh:nh + crop_shape[0], nw:nw + crop_shape[1]]
        # ret.append(mask)
        return ret[0],mask
    return ret[0]

def random_crop2(image, image2, crop_shape, mask=None):
    image_shape = image.shape
    n = (image_shape[0]+crop_shape[0]-1)//crop_shape[0]
    m = (image_shape[1]+crop_shape[1]-1)//crop_shape[1]
    if crop_shape[0]<image_shape[0]: # 当测试设定不裁剪时，略去裁剪步骤
        nh = np.random.randint(0, n)
        nw = np.random.randint(0, m)
        nh1 = nh*crop_shape[0]
        nh2 = (nh+1)*crop_shape[0]
        nw1 = nw*crop_shape[1]
        nw2 = (nw+1)*crop_shape[1]
        if nh == n-1:
            nh1=image_shape[0]-crop_shape[0]
            nh2=image_shape[0]
        if nw == m-1:
            nw1=image_shape[1]-crop_shape[1]
            nw2=image_shape[1]
        image = image[nh1:nh2, nw1:nw2]
    if mask is not None:
        if crop_shape[0] < image_shape[0]:
            mask = mask[nh1:nh2, nw1:nw2]
            image2 = image2[nh1:nh2, nw1:nw2]
        # ret.append(mask)
        return image,image2,mask
    return image,image2

def randomHorizontalFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    return image, mask

def randomHorizontalFlip(image, mask, erase=None, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)
        if erase is  not None:
            erase = cv2.flip(erase, 1)
    if erase is  not None:
        return image, mask, erase
    return image, mask

def randomVerticleFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)

    return image, mask

def randomVerticleFlip(image, mask, erase=None,  u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)
        if erase is  not None:
            erase = cv2.flip(erase, 0)
    if erase is  not None:
        return image, mask, erase
    return image, mask

def randomRotate90(image, mask,u=0.5):
    if np.random.random() < u:
        image=np.rot90(image)
        mask=np.rot90(mask)

    return image, mask

def randomRotate90(image, mask, erase=None, u=0.5):
    if np.random.random() < u:
        image=np.rot90(image)
        mask=np.rot90(mask)
        if erase is  not None:
            erase=np.rot90(erase)
    if erase is  not None:
        return image, mask, erase
    return image, mask

def randomTranslation(image, mask, u=0.2):
    rd = np.random.random()
    if rd < u:
        new_image = np.zeros((768, 768, 3), dtype=np.uint8)
        new_mask = np.zeros((768, 768), dtype=np.uint8)
        if rd < u/2:
            # 向右平移
            new_image[:512, 256:512+256] = image
            image = new_image[:512,:512:,:]
            new_mask[:512, 256:512+256] = mask
            mask = new_mask[:512,:512]
        elif rd < u:
            # 向左平移
            new_image[:512,:512] = image
            image = new_image[:512,256:512+256,:]
            new_mask[:512,:512] = mask
            mask = new_mask[:512,256:512+256]
        # elif rd < u*3/4:
        #     # 向上平移
        #     new_image[:512,:512] = image
        #     image = new_image[256:512+256, :512,:]
        #     new_mask[:512,:512] = mask
        #     mask = new_mask[256:512+256, :512]
        # else:
        #     # 向下平移
        #     new_image[256:512+256, :512] = image
        #     image = new_image[:512,:512,:]
        #     new_mask[256:512+256, :512] = mask
        #     mask = new_mask[:512,:512]
    return image, mask

def default_loader(id, root ,type=None):
    img = cv2.imread(os.path.join(root,'{}_sat.jpg').format(id))   
    img = cv2.resize(img,setting.setting['crop_size'])  
    if type=='Test':
        np.random.seed(setting.setting['seed'])
        img=random_crop(img, setting.setting['crop_size'], None)
    if type=='Train':
        mask = cv2.imread(os.path.join(root,'{}_mask.png').format(id), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask,setting.setting['crop_size'])
        img, mask = random_crop(img, setting.setting['crop_size'], mask)
        img, mask = randomHorizontalFlip(img, mask)
        img, mask = randomVerticleFlip(img, mask)
        img, mask = randomRotate90(img, mask)
        mask = np.expand_dims(mask, axis=2)
        return img, mask
    #mask = abs(mask-1)
    return img,None

def getCorruptRoad(road_gt, height, width, artifacts_shape="linear", element_counts=16):
        angle_theta = 5
        # False Negative Mask
        FNmask = np.ones((height, width), np.float32)
        # False Positive Mask
        FPmask = np.zeros((height, width), np.float32)
        indices = np.where(road_gt[0] == 1)

        if artifacts_shape == "square":
            shapes = [[16, 16], [32, 32]]
            ##### FNmask
            if len(indices[0]) == 0:  ### no road pixel in GT
                pass
            else:
                for c_ in range(element_counts):
                    c = np.random.choice(len(shapes), 1)[
                        0
                    ]  ### choose random square size
                    shape_ = shapes[c]
                    ind = np.random.choice(len(indices[0]), 1)[
                        0
                    ]  ### choose a random road pixel as center for the square
                    row = indices[0][ind]
                    col = indices[1][ind]

                    FNmask[
                        row - shape_[0] / 2 : row + shape_[0] / 2,
                        col - shape_[1] / 2 : col + shape_[1] / 2,
                    ] = 0
            #### FPmask
            for c_ in range(element_counts):
                c = np.random.choice(len(shapes), 2)[0]  ### choose random square size
                shape_ = shapes[c]
                row = np.random.choice(height - shape_[0] - 1, 1)[
                    0
                ]  ### choose random pixel
                col = np.random.choice(width - shape_[1] - 1, 1)[
                    0
                ]  ### choose random pixel
                FPmask[
                    row - shape_[0] / 2 : row + shape_[0] / 2,
                    col - shape_[1] / 2 : col + shape_[1] / 2,
                ] = 1

        elif artifacts_shape == "linear":
            ##### FNmask
            if len(indices[0]) == 0:  ### no road pixel in GT
                pass
            else:
                element_counts = 8
                for c_ in range(element_counts):
                    angle_theta = random.randrange(5,20)
                    c1 = np.random.choice(len(indices[0]), 1)[0]  ### choose random 2 road pixels to draw a line
                    c2 = np.random.choice(len(indices[1]), 1)[0]
                    cv2.line(FNmask,(indices[1][c1], indices[0][c1]),
                        (indices[1][c2], indices[0][c2]),
                        0,angle_theta * 2,
                    )
            #### FPmask
            element_counts = 3
            for c_ in range(element_counts):
                angle_theta = random.randrange(2,5)
                lenth = random.randrange(10,20)
                
                row1 = np.random.choice(height, 1)
                col1 = np.random.choice(width, 1)
                row2, col2 = (
                    row1 + np.random.choice(lenth, 1),
                    col1 + np.random.choice(lenth, 1),
                )
                cv2.line(FPmask, (col1[0], row1[0]), (col2[0], row2[0]), 1, angle_theta*2)
        erased_gt = (road_gt * FNmask) + FPmask
        erased_gt[erased_gt > 0] = 1
        return erased_gt
    
class DPGlobe_Dataset(data.Dataset):

    def __init__(self, root, type=None):
        '''

        :param trainlist:
        :param root:
        '''
        imagelist = filter(lambda x: x.find('jpg') != -1, os.listdir(root))
        imglist = list(map(lambda x: x[:-8], imagelist))
        self.ids = imglist
        self.loader = default_loader
        self.root = root
        self.type = type
        self.mean = np.array([0.5, 0.5, 0.5]).reshape(3,1,1)
        self.std = np.array([0.5, 0.5, 0.5]).reshape(3,1,1)
    def __getitem__(self, index):
        id = self.ids[index]
        img, mask = self.loader(id, self.root,type=self.type)
        img = (np.array(img, np.float32).transpose(2, 0, 1) / 255.0 -self.mean)/self.std
        img = torch.Tensor(img)
        if self.type=='Train':
            mask = np.array(mask, np.float32).transpose(2, 0, 1) / 255.0
            mask[mask > 0.5] = 1
            mask[mask <= 0.5] = 0
            c,h,w = mask.shape
            mask = torch.Tensor(mask)
            return img, mask, self.ids[index]
            
        return img

    def __len__(self):
        return len(list(self.ids))


class JHWV2_Dataset(data.Dataset):
    '''
    root
    ---img
    ---label
    数据中包含所有数据，虽打乱，但输出的顺序相同
    '''
    def __init__(self, root, type=None):
        self.shape = setting.setting['crop_size']
        self.root = root
        self.imglist = os.listdir(root + '/img')
        self.imglist.sort()
        random.shuffle(self.imglist)
        self.labellist = list(map(lambda x: x.replace('img', 'label'), self.imglist))
        self.type=type
        # self.mean = np.array([0.5181, 0.4532, 0.3734]).reshape(3, 1, 1)
        # self.std = np.array([0.0719, 0.0806, 0.1009]).reshape(3, 1, 1)
        self.mean = np.array([0.5, 0.5, 0.5]).reshape(3,1,1)
        self.std = np.array([0.5, 0.5, 0.5]).reshape(3,1,1)
        self.size = len(self.imglist)
    def __getitem__(self, index):
        img = cv2.imread(os.path.join(self.root + '/img', self.imglist[index]))
        mask = cv2.imread(os.path.join(self.root + '/label', self.labellist[index]), cv2.IMREAD_GRAYSCALE)
        if self.type == 'Test':
            # label = label*255
            np.random.seed(setting.setting['seed'])
        # img = cv2.resize(img,self.shape)
        # mask = cv2.resize(mask,self.shape)
        img, mask = random_crop(img, self.shape, mask)
        if self.type=='Train':
            # img, mask = random_crop(img, self.shape, mask)
            img, mask = randomHorizontalFlip(img, mask)
            img, mask = randomVerticleFlip(img, mask)
            img, mask = randomRotate90(img, mask)
            # img, mask = randomTranslation(img, mask)
            # img = add_gaussian_noise(image=img)
            # img = add_salt_and_pepper_noise(image=img)
            # img = apply_gaussian_blur(image=img)
        mask = np.expand_dims(mask, axis=2)
        img = (np.array(img, np.float32).transpose(2, 0, 1) / 255.0 -self.mean)/self.std
        mask = np.array(mask, np.float32).transpose(2, 0, 1) / (255.0 if mask.max()>1 else 1.0)
        mask[mask >= 0.5] = 1
        mask[mask <= 0.5] = 0
        img = torch.Tensor(img)
        mask = torch.Tensor(mask)
        return img, mask, self.imglist[index]

    def __len__(self):
        return len(self.imglist)
    

class JHWV2_Dataset_x(data.Dataset):
    '''
    root
    ---img
    ---label
    数据中包含所有数据，虽打乱，但输出的顺序相同
    '''
    def __init__(self, root, type=None):
        self.shape = setting.setting['crop_size']
        self.root = root
        self.imglist = os.listdir(root + '/H')
        self.imglist.sort()
        random.shuffle(self.imglist)
        self.imglist2 = list(map(lambda x: x.replace('H', 'L'), self.imglist))
        self.labellist = list(map(lambda x: x.replace('H', 'label'), self.imglist))
        self.type=type
        self.mean = np.array([0.5]).reshape(1,1,1)
        self.std = np.array([0.5]).reshape(1,1,1)
        self.size = len(self.imglist)
    def __getitem__(self, index):
        img = cv2.imread(os.path.join(self.root + '/H', self.imglist[index]), cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(os.path.join(self.root + '/L', self.imglist2[index]), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(os.path.join(self.root + '/label', self.labellist[index]), cv2.IMREAD_GRAYSCALE)
        if self.type == 'Test':
            np.random.seed(setting.setting['seed'])
        # img, mask, img2 = random_crop2(img, img2, self.shape, mask)
        if self.type=='Train':
            img, mask, img2 = randomHorizontalFlip(img, mask, img2)
            img, mask, img2 = randomVerticleFlip(img, mask, img2)
            img, mask, img2 = randomRotate90(img, mask, img2)
        mask = np.expand_dims(mask, axis=2)
        img = np.expand_dims(img, axis=2)
        img2 = np.expand_dims(img2, axis=2)
        img = (np.array(img, np.float32).transpose(2, 0, 1) / 255.0 -self.mean)/self.std
        img2 = (np.array(img2, np.float32).transpose(2, 0, 1) / 255.0 -self.mean)/self.std
        mask = np.array(mask, np.float32).transpose(2, 0, 1) / (255.0 if mask.max()>1 else 1.0)
        mask[mask >= 0.5] = 1
        mask[mask <= 0.5] = 0
        img = torch.Tensor(img)
        img2 = torch.Tensor(img2)
        mask = torch.Tensor(mask)
        return img, img2, mask, self.imglist[index]

    def __len__(self):
        return len(self.imglist)
    
    
# class JHWV2a_Dataset(data.Dataset):
#     '''
#     root
#     ---img
#     ---label
#     数据中包含所有数据，虽打乱，但输出的顺序相同
#     '''
#     def __init__(self, root, type=None):
#         self.shape = setting.setting['crop_size']
#         self.root = root
#         self.imglist = os.listdir(root + '/img')
#         self.imglist.sort()
#         random.shuffle(self.imglist)
#         self.labellist = list(map(lambda x: x.replace('img', 'label'), self.imglist))
#         self.type=type
#         # self.mean = np.array([0.5181, 0.4532, 0.3734]).reshape(3, 1, 1)
#         # self.std = np.array([0.0719, 0.0806, 0.1009]).reshape(3, 1, 1)
#         self.mean = np.array([0.5, 0.5, 0.5]).reshape(3,1,1)
#         self.std = np.array([0.5, 0.5, 0.5]).reshape(3,1,1)
#         self.size = len(self.imglist)
        
#     def getOrientationGT(self, keypoints, height, width):
#         vecmap, vecmap_angles = affinity_utils.getVectorMapsAngles(
#             (height, width), keypoints, theta=self.angle_theta, bin_size=10
#         )
#         vecmap_angles = torch.from_numpy(vecmap_angles)

#         return vecmap_angles
#     def __getitem__(self, index):
#         img = cv2.imread(os.path.join(self.root + '/img', self.imglist[index]))
#         mask = cv2.imread(os.path.join(self.root + '/label', self.labellist[index]), cv2.IMREAD_GRAYSCALE)
#         if self.type == 'Test':
#             # label = label*255
#             np.random.seed(setting.setting['seed'])
#         img = cv2.resize(img,self.shape)
#         mask = cv2.resize(mask,self.shape)
#         # img, mask = random_crop(img, self.shape,label)
#         if self.type=='Train':
#             img, mask = random_crop(img, self.shape, mask)
#             img, mask = randomHorizontalFlip(img, mask)
#             img, mask = randomVerticleFlip(img, mask)
#             img, mask = randomRotate90(img, mask)
#         mask = np.expand_dims(mask, axis=2)
#         img = (np.array(img, np.float32).transpose(2, 0, 1) / 255.0 -self.mean)/self.std
#         mask = np.array(mask, np.float32).transpose(2, 0, 1) / (255.0 if mask.max()>1 else 1.0)
#         mask[mask >= 0.5] = 1
#         mask[mask <= 0.5] = 0
#         # Create Orientation Ground Truth
#         keypoints = affinity_utils.getKeypoints(
#             mask, is_gaussian=False, smooth_dist=4
#         )
#         vecmap_angle = self.getOrientationGT(
#             keypoints,
#             height=self.shape,
#             width=self.shape),
#         img = torch.Tensor(img)
#         mask = torch.Tensor(mask)
#         vecmap_angle = torch.Tensor(vecmap_angle)
#         return img, mask, vecmap_angle, self.imglist[index]

#     def __len__(self):
#         return len(self.imglist)

class JLTest_Dataset(data.Dataset):
    '''
    root
    ---img
    ---label
    数据中包含所有数据，虽打乱，但输出的顺序相同
    '''
    def __init__(self, root, type=None):
        self.shape = setting.setting['crop_size']
        self.root = root
        self.imglist = os.listdir(root + '/img')
        self.imglist.sort()
        random.shuffle(self.imglist)
        # self.labellist = list(map(lambda x: x.replace('img', 'label'), self.imglist))
        self.type=type
        # self.mean = np.array([0.5181, 0.4532, 0.3734]).reshape(3, 1, 1)
        # self.std = np.array([0.0719, 0.0806, 0.1009]).reshape(3, 1, 1)
        self.mean = np.array([0.5, 0.5, 0.5]).reshape(3,1,1)
        self.std = np.array([0.5, 0.5, 0.5]).reshape(3,1,1)
        self.size = len(self.imglist)
    def __getitem__(self, index):
        img = cv2.imread(os.path.join(self.root + '/img', self.imglist[index]))
        if self.type == 'Test':
            np.random.seed(setting.setting['seed'])
        img = random_crop(img, self.shape, None)
        img = (np.array(img, np.float32).transpose(2, 0, 1) / 255.0 -self.mean)/self.std
        img = torch.Tensor(img)
        return img, self.imglist[index]

    def __len__(self):
        return len(self.imglist)

class Mass_Dataset(data.Dataset):
    '''
    root
    ---img
    ---label
    数据中包含所有数据，虽打乱，但输出的顺序相同
    '''
    def __init__(self, root, type=None):
        self.shape = setting.setting['crop_size']
        self.root = root
        self.imglist = os.listdir(root + '/image')
        self.imglist.sort()
        random.shuffle(self.imglist)
        self.labellist = list(map(lambda x: x[:-1], self.imglist))
        self.type=type
        self.mean = np.array([0.5, 0.5, 0.5]).reshape(3,1,1)
        self.std = np.array([0.5, 0.5, 0.5]).reshape(3,1,1)

    def __getitem__(self, index):
        img = cv2.imread(os.path.join(self.root + '/image', self.imglist[index]))
        label = cv2.imread(os.path.join(self.root + '/label', self.labellist[index]))[:,:,0]
        if type == 'Test': #测试时，每次裁剪相同区域
            np.random.seed(setting.setting['seed'])
        img, mask = random_crop(img, self.shape, label)
        if type=='Train':
            img, mask = randomHorizontalFlip(img, mask)
            img, mask = randomVerticleFlip(img, mask)
            img, mask = randomRotate90(img, mask)
        mask = np.expand_dims(mask, axis=2)
        # img = np.array(img, np.float32).transpose(2, 0, 1) / img.max() * 3.2 - 1.6
        img = (np.array(img, np.float32).transpose(2, 0, 1) / 255.0-self.mean)/self.std
        mask = np.array(mask, np.float32).transpose(2, 0, 1) / 1.0
        mask[mask >= 0.5] = 1
        mask[mask <= 0.5] = 0
        img = torch.Tensor(img)
        mask = torch.Tensor(mask)
        return img, mask
    
    def __len__(self):
        return len(self.imglist)

# class  Con_Dataset(data.Dataset):
#     '''
#     root
#     ---img
#     ---label
#     数据中包含所有数据，虽打乱，但输出的顺序相同
#     '''
#     def __init__(self, root, type=None):
#         self.shape = setting.setting['crop_size']
#         self.root = root
#         self.imglist = os.listdir(root + '/img')
#         self.imglist.sort()
#         random.shuffle(self.imglist)
#         self.labellist = self.imglist
#         self.type=type
#         self.mean = np.array([0.5, 0.5, 0.5]).reshape(3,1,1)
#         self.std = np.array([0.5, 0.5, 0.5]).reshape(3,1,1)
#         self.size = len(self.imglist)
#     def __getitem__(self, index):
#         img = cv2.imread(os.path.join(self.root + '/img', self.imglist[index])) 
#         mask = None
#         if os.path.isdir(self.root + '/label'):        
#             mask = cv2.imread(os.path.join(self.root + '/label', self.labellist[index]), cv2.IMREAD_GRAYSCALE)
#         if self.type == 'Test':
#             gt=1
#             if mask is not None:
#                 gt = mask
#                 gt = cv2.resize(gt,self.shape)
#                 gt = np.expand_dims(gt, axis=2)
#                 gt = np.array(gt, np.float32).transpose(2, 0, 1) / (255.0 if gt.max()>1 else 1.0)
#                 gt[gt >= 0.5] = 1
#                 gt[gt <= 0.5] = 0
#                 gt = torch.Tensor(gt)
#             mask = cv2.imread(os.path.join(self.root + '/erase', self.labellist[index]), cv2.IMREAD_GRAYSCALE)
#             np.random.seed(setting.setting['seed'])
            
#         img, mask = random_crop(img, self.shape, mask)
#         if self.type=='Train':
#             img, mask = randomHorizontalFlip(img, mask)
#             img, mask = randomVerticleFlip(img, mask)
#             img, mask = randomRotate90(img, mask)
#         mask = np.expand_dims(mask, axis=2)
#         img = (np.array(img, np.float32).transpose(2, 0, 1) / 255.0 -self.mean)/self.std
#         mask = np.array(mask, np.float32).transpose(2, 0, 1) / (255.0 if mask.max()>1 else 1.0)
#         mask[mask >= 0.5] = 1
#         mask[mask <= 0.5] = 0
#         _,h,w = mask.shape
#         erased_gt = getCorruptRoad(mask, h, w)
#         erased_gt = torch.from_numpy(erased_gt)
#         img = torch.Tensor(img)
#         mask = torch.Tensor(mask)
#         erased_gt = torch.Tensor(erased_gt)
#         if self.type == 'Test' and (not isinstance(gt,int)):   
#              return img, gt, mask, self.imglist[index]
#         return img, mask, erased_gt, self.imglist[index]

#     def __len__(self):
#         return len(self.imglist)
    

class  Con_Dataset(data.Dataset):
    '''
    root
    ---img
    ---label
    数据中包含所有数据，虽打乱，但输出的顺序相同
    '''
    def __init__(self, root, type=None):
        self.shape = setting.setting['crop_size']
        self.root = root
        self.imglist = os.listdir(root + '/img')
        self.imglist.sort()
        random.shuffle(self.imglist)
        self.labellist = self.imglist
        self.type=type
        self.mean = np.array([0.5, 0.5, 0.5]).reshape(3,1,1)
        self.std = np.array([0.5, 0.5, 0.5]).reshape(3,1,1)
        self.size = len(self.imglist)
    def __getitem__(self, index):
        img = cv2.imread(os.path.join(self.root + '/img', self.imglist[index])) 
        mask = None
        erase=None
        if os.path.isdir(self.root + '/label'):        
            mask = cv2.imread(os.path.join(self.root + '/label', self.labellist[index]), cv2.IMREAD_GRAYSCALE)
        if os.path.isdir(self.root + '/erase'): 
            erase = cv2.imread(os.path.join(self.root + '/erase', self.labellist[index]), cv2.IMREAD_GRAYSCALE)
            np.random.seed(setting.setting['seed'])
            
        img, mask = random_crop(img, self.shape, mask)
        if self.type=='Train' or 1==1:
            img, mask, erase = randomHorizontalFlip(img, mask, erase)
            img, mask, erase = randomVerticleFlip(img, mask, erase)
            img, mask, erase = randomRotate90(img, mask, erase)
            mask = np.expand_dims(mask, axis=2)
            mask = np.array(mask, np.float32).transpose(2, 0, 1) / (255.0 if mask.max()>1 else 1.0)
            mask[mask >= 0.5] = 1
            mask[mask <= 0.5] = 0
            mask = torch.Tensor(mask)
        erase = np.expand_dims(erase, axis=2)
        erase = np.array(erase, np.float32).transpose(2, 0, 1) / (255.0 if erase.max()>1 else 1.0)
        erase[erase >= 0.5] = 1
        erase[erase <= 0.5] = 0
        img = (np.array(img, np.float32).transpose(2, 0, 1) / 255.0 -self.mean)/self.std
        img = torch.Tensor(img)
        erase = torch.Tensor(erase)
        return img, mask, erase, self.imglist[index]

    def __len__(self):
        return len(self.imglist)