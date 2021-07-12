from PIL import Image
import torchvision
from torch import nn
import torch

# 图片预处理
t = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

img = Image.open('imagefolder/train/Japanese/3.jpg')        # 打开图片
img = t(img)
img = torch.reshape(img, (1, 3, 224, 224))

# 设置模型
vgg16 = torchvision.models.vgg16_bn(pretrained=False)
vgg16.classifier.add_module('last1', nn.ReLU(inplace=True))
vgg16.classifier.add_module('last2', nn.Dropout(p=0.2, inplace=False))
vgg16.classifier.add_module('last3', nn.Linear(1000, 2))
vgg16.classifier.add_module('last4', nn.Sigmoid())

vgg16.load_state_dict(torch.load('model/epoch_50.pth'))     # 加载模型参数

output = vgg16(img)                                         # 输出
if output.argmax(1).item() == 0:
    print('chinese')
else:
    print('japanese')