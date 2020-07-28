import torch.nn as nn
import torch

class MyNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.cnn_layer = nn.Sequential(
            # 224 x 224
            nn.Conv2d(3,16,3,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16,32,3,1),
            nn.ReLU(True),
            nn.MaxPool2d(2,2),#110
            nn.Conv2d(32,64,3,1),#108
            nn.ReLU(True),
            nn.MaxPool2d(2,2),#54
            nn.Conv2d(64,128,3,1),#52
            nn.ReLU(True),
            nn.MaxPool2d(2,2),#26
            nn.Conv2d(128,256,3,1),#24
            nn.ReLU(True),
            nn.AvgPool2d(2,2), #12
            nn.Conv2d(256,64,3,1),#10
            nn.ReLU(True)
        )

        self.cnn_layer2 = nn.Sequential(
            nn.Conv2d(64,128,10,1),
            nn.ReLU(True),
            nn.Conv2d(128,5,1,1)
        )
        # self.mlp_layer = nn.Sequential(
        #     nn.Linear(10*10*64, 128),
        #     nn.ReLU(),
        #     nn.Linear(128,5)
        # )


    def forward(self, x):
        x = self.cnn_layer(x)
        # x = x.reshape(-1, 10*10*64)
        # x = self.mlp_layer(x)
        x = self.cnn_layer2(x)
        # 106 5 1 1
        # 106 5
        x = x.squeeze()
        category = torch.sigmoid(x[:,0])
        axes = torch.relu(x[:, 1:])# 底层都是调的torch.rule

        return category, axes


