import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

# 设置图片处理方式
t1 = transforms.Compose([
    transforms.Resize((224, 224)),              # vgg16传入图片大小是224*224
    transforms.RandomHorizontalFlip(),          # 随机水平翻转
    transforms.RandomRotation(0.2),             # 随机旋转
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],       # 归一化到（-1,1）
                         [0.5, 0.5, 0.5])
])

# 加载原始数据集
train_dataset = datasets.ImageFolder('imagefolder/train', transform=t1)
test_dataset = datasets.ImageFolder('imagefolder/test', transform=t1)

train_loader = DataLoader(train_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)

# 设置模型
vgg16 = torchvision.models.vgg16_bn(pretrained=True)
vgg16.classifier.add_module('last1', nn.ReLU(inplace=True))
vgg16.classifier.add_module('last2', nn.Dropout(p=0.2, inplace=False))
vgg16.classifier.add_module('last3', nn.Linear(1000, 2))
vgg16.classifier.add_module('last4', nn.Sigmoid())

# 冻结层为features
for param in vgg16.features.parameters():
    param.requires_grad = False

# 使用gpu训练
vgg16.cuda()
features = vgg16.features

# 定义预加载features层处理过的数据的函数
def preconvfeat(dataset, model):
    conv_features = []
    labels_list = []
    for data in dataset:
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()
        inputs, labels = torch.autograd.Variable(inputs), torch.autograd.Variable(labels)
        output = model(inputs)
        conv_features.extend(output.data.cpu().numpy())
        labels_list.extend(labels.data.cpu().numpy())
    conv_features = np.concatenate([[feat] for feat in conv_features])
    return (conv_features,labels_list)
conv_feat_train, labels_train = preconvfeat(train_loader, features)
conv_feat_test, labels_test = preconvfeat(test_loader,features)

# 重写数据集类
class My_dataset(Dataset):
    def __init__(self, feat, labels):
        self.conv_feat = feat
        self.labels = labels

    def __len__(self):
        return len(self.conv_feat)
    def __getitem__(self, idx):
        return self.conv_feat[idx], self.labels[idx]

# 加载处理过的数据集
train_feat_dataset = My_dataset(conv_feat_train, labels_train)
test_feat_dataset = My_dataset(conv_feat_test, labels_test)

train_feat_loader = DataLoader(train_feat_dataset, batch_size=16, shuffle=True)
test_feat_loader = DataLoader(test_feat_dataset, batch_size=16, shuffle=True)

# 设置优化器
optimizer = torch.optim.Adam(vgg16.classifier.parameters(), lr=0.000015)

# 设置损失函数
loss_fn = torch.nn.CrossEntropyLoss()
loss_fn.to(torch.device('cuda'))

# 设置Flatten()
f = nn.Flatten()

# 设置图像显示
writer = SummaryWriter('logs')

# 定义fit函数
def fit(epoch, model, data_loader, phase = 'train', volatile = False):
    if phase == 'train':
        model.train()
    if phase == 'test':
        model.eval()
        volatile = True
    running_loss = 0.0
    running_correct = 0
    for batch_index, (data, target) in enumerate(data_loader):
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile), Variable(target)
        if phase == 'train':
            optimizer.zero_grad()
        data = f(data)
        output = model(data)
        loss = loss_fn(output, target)
        running_loss += loss_fn(output, target).item()
        running_correct += (output.argmax(1)==target).sum()
        if phase == 'train':
            loss.backward()
            optimizer.step()
    loss = running_loss/len(data_loader.dataset)
    accuracy = 100. * running_correct/len(data_loader.dataset)
    print(f'epoch: {epoch}:{phase} loss is {loss:{6}.{2}} and {phase} accuracy is {running_correct}/{len(data_loader.dataset)}'
          f'{accuracy:{10}.{4}}')
    if phase == 'train':
        writer.add_scalar('train_loss', loss, epoch)
    if phase == 'test':
        writer.add_scalar('accuracy', accuracy, epoch)
    torch.save(vgg16.state_dict(),'D:\\py\\pytorch\\model\\epoch_{}.pth'.format(epoch))

# 设置轮数开始训练
for epoch in range(1,51):
    fit(epoch, vgg16.classifier, train_feat_loader, phase='train')
    fit(epoch, vgg16.classifier, test_feat_loader, phase='test')

writer.close()