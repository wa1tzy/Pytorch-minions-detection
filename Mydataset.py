import torch
from torch.utils.data import Dataset,DataLoader
import os
from PIL import Image
import numpy as np
from torchvision import transforms
from torchvision.transforms import ToPILImage

class MyDataset(Dataset):

    mean = torch.tensor([0.5751, 0.5365, 0.4976])
    std = torch.tensor([0.3449, 0.3392, 0.3443])
    def __init__(self,path,train=True):
        self.path = path
        self.dataset = os.listdir(self.path)
        # 按照第一个元素排序
        self.dataset.sort(key=lambda x: int(x[:x.index(".")]))
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MyDataset.mean,MyDataset.std)
        ])
        if train:
            self.dataset = self.dataset[:4000]
        else:
            self.dataset = self.dataset[4000:5000]
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        img = self.dataset[item]
        img_data = Image.open(os.path.join(self.path, img))
        img_data = self.trans(img_data)
        labels = img.split(".")
        axes = np.array(labels[1:5], dtype=np.float32) / 224
        category = np.array(labels[5:6], dtype=np.float32)
        # 拼接列表
        target = np.concatenate((axes, category))
        return img_data , target

if __name__ == '__main__':
    data = MyDataset(r"E:\Mysoft\ZONG\yellow_minions\datasets")
    loader = DataLoader(dataset=data, batch_size=5000,shuffle=True)
    x = data[0][0]
    print(x)
    x = (x * torch.tensor(MyDataset.std, dtype=torch.float32).reshape(3,1,1) + torch.tensor(MyDataset.mean,dtype=torch.float32).reshape(3,1,1))
    x = ToPILImage()(x)
    print(x)
    x.show()

    # data = next(iter(loader))[0]
    # mean = torch.mean(data,dim=(0,2,3))
    # std = torch.std(data, dim=(0,2,3))
    # print(mean , std)
    # a = np.array([0.2,0.3,0.5,0.2])
    # b = np.array([1])
    #
    # print(np.concatenate((a,b)))
