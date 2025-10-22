import torch
import torch.nn as nn

class dice_bce_loss(nn.Module):
    def __init__(self, batch=True):
        super(dice_bce_loss, self).__init__()
        self.batch = batch 
        self.bce_loss = nn.BCELoss()
           
    def __call__(self, y_true, y_pred):
        a =  self.bce_loss(y_pred, y_true)
        smooth = 1e-6  # may change

        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        b = 1- score.mean()
        return a + b