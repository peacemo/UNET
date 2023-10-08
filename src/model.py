'''
 # @ Author: Chen Xueqiang
 # @ Create Time: 2023-10-06 11:46:00
 # @ Modified by: Chen Xueqiang
 # @ Modified time: 2023-10-08 10:49:37
 # @ Description: UNET from scratch. 
 '''


import torch 
import torch.nn as nn 
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    """Every conv layer in a level (both downs and ups), there are 2 convs
       in 1 level. (Based on the paper). 
    """
    def __init__(self, in_channels, out_channels, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv = nn.Sequential(
            # 1 conv2d followed by a relu, totally 2 convs
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False), 
            nn.BatchNorm2d(out_channels), 
            nn.ReLU(inplace=True), 
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False), 
            nn.BatchNorm2d(out_channels), 
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
    
class UNET(nn.Module):
    def __init__(
            self, 
            in_channels=3, 
            out_channels=1, 
            features=[64, 128, 256, 512], 
            *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.downs = nn.ModuleList()  # downscals
        self.ups = nn.ModuleList()  # upscales
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # maxpooling for every level 

        # downs 
        for feature in features:  # 4 double_conv layers
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        self.bottom_conv = DoubleConv(features[-1], features[-1] * 2)  # bottom double_conv layers

        # ups
        for feature in reversed(features):  # 4 concat & double conv layers
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)  # final conv for output

    def forward(self, x):
        skip_connections = []

        for level in self.downs:
            x = level(x)
            skip_connections.append(x)  # save the information of every down level 
            x = self.pool(x)

        x = self.bottom_conv(x)

        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)

            skip = skip_connections[idx//2]
            if skip.shape != x.shape:
                x = TF.resize(x, size=skip.shape[2:])

            concat_skip = torch.cat((skip, x), dim=1)  # concat the information from the downs
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)
        
# def test():
#     x = torch.randn((3, 1, 161, 161))
#     unet = UNET(1, 1)
#     preds = unet(x)
#     print(f'{x.shape=}')
#     print(f'{preds.shape=}')

# test()