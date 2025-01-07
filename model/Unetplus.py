import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(Down, self).__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2,2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self,x):
        return self.down(x)

class Up(nn.Module):
    def __init__(self,in_channels, out_channels, n_concat = 2):
        super(Up, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                nn.Conv2d(in_channels,in_channels//2, kernel_size=1,padding=0),
                                nn.ReLU(inplace=True))
        self.conv = DoubleConv(in_channels+(n_concat-2)*out_channels, out_channels)

    def forward(self,x1,*x2):
        x1 = self.up(x1)
        for x in x2:
            x1 = torch.cat([x,x1],dim=1)
        return self.conv(x1)

class unetpluses(torch.nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        # column : 0
        self.x00_down = DoubleConv(in_channels,64)
        self.x10_down = Down(64,128)
        self.x20_down = Down(128,256)
        self.x30_down = Down(256,512)
        self.x40_down = Down(512,1024)

        # column : 1
        self.x01_up = Up(128, 64)
        self.x11_up = Up(256, 128)
        self.x21_up = Up(512, 256)
        self.x31_up = Up(1024, 512)

        # column : 2
        self.x02_up = Up(128, 64, 3)
        self.x12_up = Up(256, 128, 3)
        self.x22_up = Up(512, 256, 3)

        # column : 3
        self.x03_up = Up(128, 64, 4)
        self.x13_up = Up(256, 128, 4)

        # column : 4
        self.x04_up = Up(128, 64, 5)
        self.classification = nn.Conv2d(64,num_classes,kernel_size=1)

    def forward(self,x):
        x00 = self.x00_down(x)
        x10 = self.x10_down(x00)
        x20 = self.x20_down(x10)
        x30 = self.x30_down(x20)
        x40 = self.x40_down(x30)

        x01 = self.x01_up(x10,x00)
        x11 = self.x11_up(x20,x10)
        x21 = self.x21_up(x30,x20)
        x31 = self.x31_up(x40,x30)

        x02 = self.x02_up(x11,x01,x00)
        x12 = self.x12_up(x21,x11,x10)
        x22 = self.x22_up(x31,x21,x20)

        x03 = self.x03_up(x12,x02,x01,x00)
        x13 = self.x13_up(x22,x12,x11,x10)

        x04 = self.x04_up(x13,x03,x02,x01,x00)
        x = self.classification(x04)
        # x = F.sigmoid(x)

        return x

