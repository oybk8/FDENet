
import setting
import os

from tqdm import tqdm
from time import time   
import torch
from networks.FDENet import  FDENet
from framework import MyFrame
from loss import  dice_bce_loss
from data import *
torch.cuda.set_device(0)
# os.environ["CUDA_VISIBLE_DEVICES"]=setting.setting['gpu']

dataset = setting.setting['dataset']
SHAPE = setting.setting['crop_size']
dataroot = setting.setting['dataroot']
model = setting.setting['model']
NAME = setting.setting['exper_name']
BATCHSIZE_PER_CARD = setting.setting['batch_size_card']
SEED = setting.setting['seed']
total_epoch = setting.setting['epoch']
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

solver = MyFrame(FDENet, dice_bce_loss, lr=2e-4, istrain=False)

batchsize = BATCHSIZE_PER_CARD
# batchsize = (torch.cuda.device_count()) * BATCHSIZE_PER_CARD

# if dataset == 'DPGlobe':
#     train_dir = os.path.join(dataroot,'train_dv/')
#     dataset = DPGlobe_Dataset(train_dir,type='Train')
# if dataset == 'Mass':
#     dataset = Mass_Dataset(dataroot,type='Train')
#     trainset_indexs = torch.arange(len(dataset))[:int(len(dataset)*0.8)]
#     dataset = torch.utils.data.Subset(dataset, trainset_indexs)
if dataset == 'JLYH':
    train_dir = os.path.join(dataroot,'train_dv')
    dataset = JHWV2_Dataset(train_dir,type='Train')

# if os.path.exists('weights/' + NAME + '_JL.th'):# checkpoint
# solver.load('weights/FDNet_DP_0923.th')
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batchsize,
    shuffle=True,
    num_workers=4) 

mylog = open('logs/'+NAME+'.log','w')# checkpoint
tic = time()
no_optim = 0
train_epoch_best_loss = 100.
start_epoch=0
for epoch in range(start_epoch, total_epoch + 1): # checkpoint 
    data_loader_iter = iter(data_loader)
    train_epoch_loss = 0
    for img, mask, name in tqdm(data_loader_iter):
        # cv2.imwrite('_img.png', (((img[0].numpy()*std)+mean)*255).astype(np.uint8).transpose(1,2,0))
        # cv2.imwrite('_mask.png', (mask[0].numpy().astype(np.uint8)*255).transpose(1,2,0))
        solver.set_input(img, mask)
        train_loss = solver.optimize(epoch)
        train_epoch_loss += train_loss
    train_epoch_loss /= len(data_loader_iter)
    print('********', file=mylog)
    print('lr ',solver.optimizer.param_groups[0]['lr'])
    print('epoch:', epoch, '    time:', int(time() - tic), file=mylog)
    print('train_loss:', train_epoch_loss.item(), file=mylog)
    print('SHAPE:', SHAPE, file=mylog)
    print('********')
    print('epoch:', epoch, '    time:', int(time() - tic))
    print('train_loss:', train_epoch_loss.item())
    print('SHAPE:', SHAPE)
    
    if train_epoch_loss >= train_epoch_best_loss:
        no_optim += 1
    else:
        no_optim = 0
        train_epoch_best_loss = train_epoch_loss
        solver.save('weights/'+NAME+'_DP_0924.th')

    # if epoch==int(total_epoch/3):
    #     #solver.load('weights/'+NAME+'.th')
    #     # solver.save('weights/'+NAME+str(epoch)+'.th')
    #     solver.update_lr(2.0, factor = True, mylog = mylog)

    if epoch==80:
        #solver.load('weights/'+NAME+'.th')
        # solver.save('weights/'+NAME+str(epoch)+'.th')
        solver.update_lr(5.0, factor = True, mylog = mylog)
        
    if epoch==110:
        solver.save('weights/'+NAME+str(epoch)+'.th')
        #solver.load('weights/'+NAME+'.th')
        solver.update_lr(5.0, factor = True, mylog = mylog)
    
    if epoch==130:
        solver.save('weights/'+NAME+str(epoch)+'.th')
        #solver.load('weights/'+NAME+'.th')
        solver.update_lr(5.0, factor = True, mylog = mylog)
        
    mylog.flush()
    # # 早停检查
    # early_stopper(train_epoch_loss, solver.net)
    # if early_stopper.early_stop:
    #     print("Early stopping triggered!")
    #     break
    
print('Finish!', file=mylog)
print('Finish!')
mylog.close()
