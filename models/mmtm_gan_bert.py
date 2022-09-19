from .r2plus1d import r2plus1d_34_32_ig65m
from .BERT.self_attention import self_attention
from .mmtm import MMTM

import torch
import torch.nn as nn




class Generator(nn.Module):
    def __init__(self,hiddenSize):
        super(Generator,self).__init__()
        self.nLayers = 1
        self.hiddenSize = hiddenSize
        self.attnHeads = 8
        
        #self.mmtm0 = MMTM(45,45,4)
        self.rgbFeatures = nn.Sequential(*list(
            r2plus1d_34_32_ig65m(359,pretrained = True, progress = True).children())[:-2])
        print(self.rgbFeatures)

        self.depthFeatures = nn.Sequential(*list(
            r2plus1d_34_32_ig65m(359,pretrained = True, progress = True).children())[:-2])
        self.depthFeatures[0][0] = nn.Conv3d(1, 45, kernel_size=(1, 7, 7), 
                                          stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
        
        self.avgPool = nn.AdaptiveAvgPool3d((1,1,1))

    def forward(self,rgb,depth):
        #torch.Size([3, 8, 256, 256, 3])
        B,F,W,H,C = rgb.shape
        rgb = rgb.view(-1,F,3,W,H).transpose(1,2)
        rgb = self.rgbFeatures(rgb)
        
        depth = depth.view(-1,F,1,W,H).transpose(1,2)
        depth = self.depthFeatures(depth)
        rgb = self.avgPool(rgb)
        depth = self.avgPool(depth)

        #(b,hiddenSize)
        rgb = rgb.view(rgb.size(0), self.hiddenSize)
        depth = depth.view(depth.size(0), self.hiddenSize)

        return rgb,depth

class Classifer(nn.Module):
    def __init__(self,inChannals,classNum):
        super(Classifer,self).__init__()
        self.nLayers = 1
        self.hiddenSize = inChannals
        self.attnHeads = 8
        #self.self_attention = self_attention(self.hiddenSize, 2 , hidden=self.hiddenSize, 
        #                                     n_layers=self.nLayers, attn_heads=self.attnHeads)
        self.classifer = nn.Sequential(
            nn.Linear(2 * inChannals,inChannals),
            nn.LeakyReLU(inplace = True),
            nn.Dropout(0.5),
            nn.Linear(inChannals,inChannals),
            nn.LeakyReLU(inplace = True),
            nn.Dropout(0.5),
            nn.Linear(inChannals,classNum)
        )
    def forward(self,rgb,depth):
        #rgb = rgb.view(rgb.shape + (1,))
        #depth = depth.view(depth.shape + (1,))

        total = torch.cat([rgb,depth],dim = 1)#.transpose(1, 2)
        out = self.classifer(total)
        #output, maskSample = self.self_attention(total)
        #out = self.classifer(output[:,0,:])
        return out

class Discriminator(nn.Module):
    def __init__(self,inChannals,classNum,ratio = 1):
        # 1：RGB   0：Depth
        super(Discriminator,self).__init__()
        self.features = nn.Sequential(
            nn.Linear(inChannals,inChannals),
            nn.LeakyReLU(inplace = True),
            nn.Linear(inChannals,inChannals * ratio),
            nn.LeakyReLU(inplace = True),
            nn.Linear(inChannals * ratio,inChannals * ratio),
            nn.LeakyReLU(inplace = True),
            nn.Linear(inChannals * ratio,inChannals * ratio),
            nn.LeakyReLU(inplace = True)
        )


        self.discriminator = nn.Sequential(
            nn.Linear(inChannals * ratio,1)
        )
        self.classifer = nn.Sequential(
            nn.Dropout(),
            nn.Linear(inChannals * ratio,classNum)
        )

    def forward(self,input):
        input = self.features(input)
        dis = self.discriminator(input)
        out = self.classifer(input)
        return dis,out