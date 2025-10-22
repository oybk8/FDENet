import torch.nn as nn
import torch
from networks.transx import transxnet_t
from networks.SFCD import SFCD

class BasicConv2d_relu(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d_relu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self._init_weight()
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(self.bn(x))
        return x
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
class FDENet(nn.Module):
    def __init__(self):
        super(FDENet, self).__init__()
        filters = [48, 96, 224, 448] #[96, 192, 384, 768] #[48, 96, 224, 448]
        self.rgb_tranx= transxnet_t(pretrained=True)
        self.decoder = SFCD(filters)
        self.deconv = nn.Sequential(nn.ConvTranspose2d(int(filters[0]/2), int(filters[0]/4), 3, stride=2, padding=1, output_padding=1),
                                    nn.BatchNorm2d(int(filters[0]/4)),
                                    nn.ReLU(),
                                    nn.Conv2d(int(filters[0]/4), int(filters[0]/4),1, bias=False),
                                    nn.BatchNorm2d(int(filters[0]/4)),
                                    nn.ReLU()
                                    )
        self.finalconv = nn.Conv2d(int(filters[0]/4),1,1)
    def forward(self,x):
        x4 = self.rgb_tranx(x)
        x = self.decoder(x4)
        x = self.finalconv(self.deconv(x))
        return torch.sigmoid(x)