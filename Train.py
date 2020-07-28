import torch
from torch.utils import data
import torch.nn as nn
from torchvision import transforms
from Mynet import MyNet
from Mydataset import MyDataset
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
import os

class Trainer:

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MyDataset.mean,MyDataset.std)
        ])
        self.net = MyNet().to(self.device)
        self.optimier = torch.optim.Adam((self.net.parameters()))
        self.train_dataset = MyDataset(r"E:\Mysoft\ZONG\yellow_minions\datasets", train=True)
        self.test_dataset = MyDataset(r"E:\Mysoft\ZONG\yellow_minions\datasets", train=False)
        self.offset_lossfunc = nn.MSELoss().to(self.device)
        self.category_lossfunc = nn.BCELoss().to(self.device)

    def train(self):
        if os.path.exists("models/net.pth"):
            self.net = torch.load("models/net.pth")
            print("exists")
        BATCH_SIZE = 108
        NUM_EPOCHS = 20
        trainloader = data.DataLoader(dataset=self.train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        losses = []
        for epochs in range(NUM_EPOCHS):
            print("epochs:{}".format(epochs))
            for i, (x , y) in enumerate(trainloader):
                x = x.to(self.device)
                y = y.to(self.device)
                category,axes = self.net(x)
                loss1 = self.category_lossfunc(category, y[:, 4])
                loss2 = self.offset_lossfunc(axes, y[:, 0:4])  # (106,4)
                loss = loss1 + loss2
                if i % 5 == 0:
                    losses.append(loss.float())
                    print("iteration:{}/{},loss:{}".format(i, len(trainloader), loss.float()))
                    plt.clf()
                    plt.plot(losses)
                    plt.pause(0.1)
                    plt.title("loss")
                    plt.savefig("loss.jpg")
                self.optimier.zero_grad()
                loss.backward()
                self.optimier.step()
                del x, y, category, axes, loss1, loss2, loss # 清空变量，以防内存溢出
            torch.save(self.net, "models/net.pth")

    def test(self):
        testloader = data.DataLoader(dataset=self.test_dataset, batch_size=50, shuffle=True)
        net = torch.load("models/net.pth")
        total = 0
        for epochs,(x, y) in enumerate(testloader):
            x = x.to(self.device)
            y = y.to(self.device)
            category, axes = net(x)
            index = category.round() == 1 # 索引集 为有小黄人的索引，返回的是布尔值
            target = y[index] # 网络预测有小黄人对应的标签值，是为了反算到原图上
            total += (category.round() == y[:, 4]).sum().cpu().numpy() #预测对的总数
            """
            计算精确率：分别找到预测和原数据为正样本的索引值，求交集获取TP，precision=TP/预测为正样本的总数量
            计算召回率：分别找到预测和原数据为正样本的索引值，求交集获取TP，recall=TP/原数据为正样本的总数量
            """
            bool_index = y[:, 4] == 1 # 原始数据里为正样本的布尔索引
            index2 = torch.nonzero(bool_index) # 根据布尔索引获得非零元素的索引
            a_index = (index2.flatten())# 将索引平铺，为了做比较计算
            bool_Index2 = category.round() == 1  # (TP+FP) # 预测为正样本的布尔索引
            b_index = torch.nonzero(bool_Index2).flatten() # 预测为正样本的索引并平铺
            TP = np.intersect1d(a_index.cpu().numpy(), b_index.cpu().numpy())# 计算交集
            Precision = len(TP) / ((category.round() == 1).sum().cpu().numpy())*100
            Recall = len(TP) / ((y[:, 4] == 1).sum().cpu().numpy())*100
            print("epochs:{},Precision:{:.3f}%,Recall:{:.3f}%".format(epochs,Precision,Recall))

            # 将预测为正样本的数据还原
            x = (x[index].cpu() * MyDataset.std.reshape(-1, 3, 1, 1) + MyDataset.mean.reshape(-1, 3, 1, 1))
            for j, i in enumerate(axes[index]):  # (坐标还原)
                boxes = (i.data.cpu().numpy() * 224).astype(np.int32)# 预测坐标
                target_box = (target[j, 0:4].data.cpu().numpy() * 224).astype(np.int32)# 实际坐标
                img = transforms.ToPILImage()(x[j]) # 图片还原
                plt.clf()
                plt.axis("off")
                draw = ImageDraw.Draw(img)
                draw.rectangle(boxes.tolist(), outline="red")  # 预测值
                draw.rectangle(target_box.tolist(), outline="yellow")  # 原始值
                plt.imshow(img)
                plt.pause(1)
                del boxes, target_box, img, draw
            del x, y, category, axes, index, target

        print("正确率:", total/len(self.test_dataset))
if __name__ == '__main__':
    t = Trainer()
    # t.train()
    t.test()