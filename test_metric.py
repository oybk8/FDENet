import os
import setting
os.environ["CUDA_VISIBLE_DEVICES"]= setting.setting['gpu']
import time
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from networks.FDENet import FDENet
from data import * 
import torch.nn.functional as F
torch.cuda.set_device(1)

NAME = setting.setting['exper_name']
BATCHSIZE_PER_CARD = setting.setting['batch_size_card']
SEED = setting.setting['seed']
out_pred = setting.setting['out_pred']
dataroot = setting.setting['dataroot']
dataset = setting.setting['dataset']
model = setting.setting['model']
shape = setting.setting['crop_size']

def tensor_bound(img, k_size):
    B, C, H, W = img.shape
    pad = int((k_size - 1) // 2)
    img_pad = F.pad(img, pad=[pad, pad, pad, pad], mode='constant', value=0)
    # unfold in the second and third dimensions
    patches = img_pad.unfold(2, k_size, 1).unfold(3, k_size, 1)
    corrosion, _ = torch.min(patches.contiguous().view(B, C, H, W, -1), dim=-1)
    inflation, _ = torch.max(patches.contiguous().view(B, C, H, W, -1), dim=-1)
    return inflation - corrosion

target = './results/'+NAME+'_'+dataset+'/'
# target = '/storage/mobile/BK/code/OARENet-main/dataset/train_cpl_dp/'+dataset+'/'

# if dataset == 'DPGlobe':
#     test_dir = os.path.join(dataroot,'SwinT_OA_JL_RN_AS_DP/')
#     dataset = DPGlobe_Dataset(test_dir,type='Train')
# if dataset == 'JHWV2':
#     dataset = JHWV2_Dataset(dataroot,type='Test')
#     # testset_indexs = torch.arange(len(dataset))[int(len(dataset)*0.8):]
#     # dataset = torch.utils.data.Subset(dataset, testset_indexs)
# if dataset == 'Mass':
#     dataset = Mass_Dataset(dataroot,type='Test')
#     trainset_indexs = torch.arange(len(dataset))[int(len(dataset)*0.8):]
#     dataset = torch.utils.data.Subset(dataset, trainset_indexs)
    
if dataset == 'JLYH':
    train_dir = os.path.join(dataroot,'valid_dv') # valid_dv
    dataset = JHWV2_Dataset(train_dir,type='Test')

mean = np.array([0.5, 0.5, 0.5]).reshape(3, 1, 1)
std = np.array([0.5, 0.5, 0.5]).reshape(3, 1, 1)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

if not os.path.isdir(target):
    os.makedirs(target)
net = FDENet()
net = net.cuda()
# net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
# net.load_state_dict(torch.load('./weights/'+NAME+'.th'),strict=False)
checkp=torch.load('weights/FDNet_JLYH_0924.th', map_location=torch.device('cpu'))
# state_dict = {k.replace('module.', ''): v for k, v in checkp.items()} 
# state_dict = {k.replace('conv4.', ''): v for k, v in state_dict.items()} 
net.load_state_dict(checkp,strict=False)

batchsize =  BATCHSIZE_PER_CARD*5  #torch.cuda.device_count() *

data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=2,
    shuffle=False,
    num_workers=8)

tic = time.time()
class SegmentationMetric():
    def __init__(self,numclass):
        self.numclass = numclass
        self.confusionMatrix = np.zeros((self.numclass,) * 2)

    def CM(self, mask, pred):
        mask_num = mask.numpy().astype('int')
        pred = pred.astype('int')
        pred = pred.flatten()
        mask = mask_num.flatten()
        cm = confusion_matrix(mask,pred)
        self.confusionMatrix += cm
        # return cm

    def PA(self):
        # PA = acc
        acc = np.diag(self.confusionMatrix).sum()/self.confusionMatrix.sum()
        return acc

    def CPA(self):
        # CPA = (tp)/tp+fp
        # precision
        cpa = np.diag(self.confusionMatrix)/self.confusionMatrix.sum(axis=0)
        return cpa

    def meanPA(self):
        cpa = self.CPA()
        
        meanpa = np.nanmean(cpa)
        return meanpa
    def CPR(self):
        # CPR:class recall
        cpr = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return cpr

    def IOU(self):
        intersection = np.diag(self.confusionMatrix)  
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(self.confusionMatrix)
        iou = intersection/union
        return iou
    
    def MIOU(self):
        intersection = np.diag(self.confusionMatrix)  
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(self.confusionMatrix)
        iou = intersection/union
        miou = 0.5*iou[0]+0.5*iou[1]
        return miou
metric = SegmentationMetric(2)
# cpa1=[]
# cpr1=[]
# f11=[]
# iou1=[]
title = 0
net.eval()
with torch.no_grad():
    for data_loader_iter in tqdm(iter(data_loader)):
        img,mask,name= data_loader_iter
        # img,name= data_loader_iter
        img = img.cuda()
        # print(summary(net,img))
        # pred1,pred2,pred3= net(img)
        preds= net(img)
        pred = preds
        # pred2 = preds[1]
        # pred=pred1
        pred[pred > 0.5] = 1
        pred[pred <= 0.5] = 0
        # pred2[pred2 > 0.5] = 1
        # pred2[pred2 <= 0.5] = 0
        # pred3[pred3 > 0.5] = 1
        # pred3[pred3 <= 0.5] = 0
        gts_eg=tensor_bound(mask, 3).cuda()
        if out_pred==True:
            for i in range(0,pred.shape[0]):
                maskout = mask[i].cpu()
                gt_eg = gts_eg[i].cpu()
                imgout = img[i].cpu() 
                predout = pred[i].cpu()
                # cv2.imwrite(target +name[i][:-4]+'_mask.png', (maskout.numpy().astype(np.uint8)*255).transpose(1,2,0))
                # cv2.imwrite(target + name[i][:-4] + '_img.png', (((imgout.numpy()*std)+mean)*255).astype(np.uint8).transpose(1,2,0))
                cv2.imwrite(target + name[i], (predout.numpy().astype(np.uint8)*255).transpose(1,2,0))
                # cv2.imwrite(target + name[i]+ '_r.png', (pred2[i].cpu().numpy().astype(np.uint8)*255).transpose(1,2,0))
                # cv2.imwrite(target + name[i]+ '_t.png', (pred3[i].cpu().numpy().astype(np.uint8)*255).transpose(1,2,0))
                # cv2.imwrite(target + name[i] + '_pred.png', (predout.numpy().astype(np.uint8)*255).transpose(1,2,0))
                # cv2.imwrite(target + name[i], (predout.numpy().astype(np.uint8)).transpose(1,2,0))
                title += 1
        pred = pred.squeeze().cpu().data.numpy()
        # mask.shape(3,1,768,768) pred.shape(3,768,768)
        metric.CM(mask, pred)
cpa = metric.CPA()[1]
cpr = metric.CPR()[1]
f1 = 2*(cpa*cpr/(cpa+cpr))
iou = metric.IOU()[1]
miou = metric.MIOU()
# 写入实验结果
metric_file = open('./logs/' + NAME + '_metric.log', 'a')
# 写入实验条件
metric_file.write('--------%s--------\n' % str(time.strftime('%Y-%m-%d %H:%M',time.localtime(time.time()))))
for k,v in setting.setting.items():
    metric_file.write('{}:{}\n'.format(k, v))
metric_file.write('cpa:{}\n'.format(cpa))
metric_file.write('cpr:{}\n'.format(cpr))
metric_file.write('f1:{}\n'.format(f1))
metric_file.write('iou:{}\n'.format(iou))
metric_file.write('miou:{}\n'.format(miou))
print('Finish!')
metric_file.close()

