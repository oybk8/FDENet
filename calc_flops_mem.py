import os
import time

from networks.FDENet import FDENet
import torch
import torch.nn.functional as F
from thop import profile 
from thop import clever_format
import torchvision.models as models
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
torch.cuda.set_device(3)
# 你的输入图像，这里使用随机数生成一个示例图像
# model = SwinT_OAM().cuda()
# model.eval()

# # 生成一系列图像
# input_images = [torch.randn(1, 3, 512, 512) for _ in range(100)]

# # 记录所有帧的处理时间
# frame_times = []

# # 开始计时
# start_time = time.time()

# # 处理所有图像
# for image in input_images:
#     model(image.cuda())
#     end_time = time.time()
#     frame_times.append(end_time - start_time)
#     start_time = end_time

# # 计算所有帧的平均处理时间
# average_time_per_frame = sum(frame_times) / len(frame_times)

# # 计算FPS
# fps = 1 / average_time_per_frame

# print(f"Average FPS: {fps}")

# # print(device)
model = FDENet()
# model = models.resnet18()

# model = torch.nn.DataParallel(model)
x = torch.randn(1, 1, 3, 512, 512)
y = torch.randn(2, 3, 512, 512)
z = torch.randn(3, 352, 352)

# x = torch.randn(1, 3, 352, 352).cuda()
# y = torch.randn(1, 3, 352, 352).cuda()
# z = torch.randn(1, 100, 352, 352).cuda()
# v = torch.randn(1, 100, 352, 352).cuda()
flops, params = profile(model, inputs=(x))
f,p = clever_format([flops, params], "%3f")
print(f,p)