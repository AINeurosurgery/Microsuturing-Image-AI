#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 20:54:46 2022

@author: rohan
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from res2net import res2net50_v1b_26w_4s





class arch1(nn.Module):
    def __init__(self):
        super().__init__()

        self.resnet = res2net50_v1b_26w_4s(pretrained=True)
        
        self.x5_dim = nn.Sequential(nn.Conv2d(2048, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.x4_dim = nn.Sequential(nn.Conv2d(1024, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.x3_dim = nn.Sequential(nn.Conv2d(512, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.x2_dim = nn.Sequential(nn.Conv2d(256, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))

        self.merge_conv = nn.Sequential(nn.Conv2d(32*4, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        
        self.fc1 = nn.Sequential(
            nn.Linear(2048,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,32),
            nn.ReLU(),
            nn.Linear(32,10),
            nn.ReLU(),
            nn.Linear(10,1),
            #nn.Sigmoid(),
            #nn.Softmax(dim=-1),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(2048,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,32),
            nn.ReLU(),
            nn.Linear(32,10),
            nn.ReLU(),
            nn.Linear(10,1),
            #nn.Sigmoid(),
            #nn.Softmax(dim=-1),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(2048,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,32),
            nn.ReLU(),
            nn.Linear(32,10),
            nn.ReLU(),
            nn.Linear(10,1),
            #nn.Sigmoid(),
            #nn.Softmax(dim=-1),
        )
        self.fc4 = nn.Sequential(
            nn.Linear(2048,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,32),
            nn.ReLU(),
            nn.Linear(32,10),
            nn.ReLU(),
            nn.Linear(10,1),
            #nn.Sigmoid(),
            #nn.Softmax(dim=-1),
        )
        self.fc5 = nn.Sequential(
            nn.Linear(2048,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,32),
            nn.ReLU(),
            nn.Linear(32,10),
            nn.ReLU(),
            nn.Linear(10,1),
            #nn.Sigmoid(),
            #nn.Softmax(dim=-1),
        )
        self.fc6 = nn.Sequential(
            nn.Linear(2048+5,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,32),
            nn.ReLU(),
            nn.Linear(32,10),
            nn.ReLU(),
            nn.Linear(10,1),
            #nn.Sigmoid(),
            #nn.Softmax(dim=-1),
        )



    def forward(self,x):
        
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        
        x = self.resnet.maxpool(x)      # bs, 64, 88, 88
        x1 = self.resnet.layer1(x)      # bs, 256, 88, 88
        x2 = self.resnet.layer2(x1)     # bs, 512, 44, 44
        x3 = self.resnet.layer3(x2)     # bs, 1024, 22, 22
        x4 = self.resnet.layer4(x3)     # bs, 2048, 11, 11
        
        x4_dim = self.x5_dim(x4)
        x3_dim = self.x4_dim(x3)
        x2_dim = self.x3_dim(x2)
        x1_dim = self.x2_dim(x1)
        
        x4_dim = F.interpolate(x4_dim,size=x1_dim.size()[2:], mode='bilinear', align_corners=True)
        x3_dim = F.interpolate(x3_dim,size=x1_dim.size()[2:], mode='bilinear', align_corners=True)
        x2_dim = F.interpolate(x2_dim,size=x1_dim.size()[2:], mode='bilinear', align_corners=True)

        merged = self.merge_conv(torch.concat([x4_dim, x3_dim, x2_dim, x1_dim], dim=1))
        
        #merged = F.dropout2d(merged, p=0.3)
        #x5 = self.l1(x4)
        #x = self.l2(x5)

        #x = self.block3_1(x5)
        #x = self.block3_2(x)
        x= merged
        x = F.adaptive_avg_pool2d(x,(8,8))
        
        #x4_dim = F.interpolate(x4_dim,size=x.size()[2:], mode='bilinear', align_corners=True)
        #x = torch.concat([x, x4_dim] , dim=1)
        x = x.reshape(x.shape[0],-1)
        
        y1 = self.fc1(x)
        y2 = self.fc2(x)
        y3 = self.fc3(x)
        y4 = self.fc4(x)
        y5 = self.fc5(x)
        y = torch.cat([x,y1,y2,y3,y4,y5],dim=1)
        y6 = self.fc6(y)

        return y6,y5,y4,y3,y2,y1

"""
def test():
    model = Archi().cpu()
    print(model)
    inp1 = torch.randn((1,3,388,175))
    #inp2 = torch.randn((1,3,128,128)).cuda()
    #inp3 = torch.randn((1,19)).cuda()

    out = model(inp1)
    print(out.shape)
    print(out)


test()
"""