import torch
from edges import Orientation, nms


r = 4
imgs = torch.ones((512, 512)).reshape(1, 1, 512, 512)
ori_func = Orientation(r)
oris = ori_func(imgs)
for i in range(imgs.shape[0]):
    img = imgs[i].squeeze()
    ori = oris[i].squeeze()
    nms(img, ori)
