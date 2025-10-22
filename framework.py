import torch
import torch.nn as nn
from torch.autograd import Variable as V
import torch.nn.functional as F
import numpy as np
torch.cuda.set_device(0)

def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6

def tensor_bound(img, k_size):
    B, C, H, W = img.shape
    pad = int((k_size - 1) // 2)
    img_pad = F.pad(img, pad=[pad, pad, pad, pad], mode='constant', value=0)
    # unfold in the second and third dimensions
    patches = img_pad.unfold(2, k_size, 1).unfold(3, k_size, 1)
    corrosion, _ = torch.min(patches.contiguous().view(B, C, H, W, -1), dim=-1)
    inflation, _ = torch.max(patches.contiguous().view(B, C, H, W, -1), dim=-1)
    return inflation - corrosion

class MyFrame():
    def __init__(self, net, loss , lr=2e-4, evalmode = False):
        self.net = net().cuda()
        device_ids=[1,2]
        self.net = torch.nn.DataParallel(self.net, device_ids=device_ids)
        self.net.train()
        self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=lr)
        # self.optimizer = torch.optim.Adam([{'params':params1, 'lr': lr*0.1},{'params':params2}], lr)
        self.loss = loss()
        self.old_lr = lr
        if evalmode:
            for i in self.net.modules():
                if isinstance(i, nn.BatchNorm2d):
                    i.eval()
        
    def set_input(self, img_batch, mask_batch=None, img_id=None):
        self.img = img_batch
        self.mask = mask_batch
        self.img_id = img_id

        
    def forward(self, volatile=False):
        self.img = V(self.img.cuda(), volatile=volatile)
        if self.mask is not None:
            self.mask = V(self.mask.cuda(), volatile=volatile)
        
    def optimize(self, epoch):
        self.forward()
        self.optimizer.zero_grad()
        pred = self.net.forward(self.img)
       
        loss = self.loss(self.mask, pred)

        loss.backward()
        self.optimizer.step()
        return loss.data
        
    def save(self, path):
        torch.save(self.net.state_dict(), path)
        
    def load(self, path):
        checkp=torch.load(path, map_location=torch.device('cpu'))
        self.net.load_state_dict(checkp, strict=False)
    
    def update_lr(self, new_lr, mylog, factor=False):
        if factor:
            new_lr = self.old_lr / new_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

        print('update learning rate: %f -> %f' % (self.old_lr, new_lr), file=mylog)
        print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
        self.old_lr = new_lr


def check_keywords_in_name(name, keywords=()):
    isin = False
    if keywords in name:
        isin = True
    return isin