import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

import numpy as np
import random as ra

from models.r2plus1d import r2plus1d_34_32_ig65m,r2plus1d_34_32_kinetics

from models.BERT.bert import BERT5


class GradReverse(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * -self.lambd)


def grad_reverse(x, lambd=1.0):
    return GradReverse(lambd)(x)

class Classifier(nn.Module):
    def __init__(self, num_class=11, inc=512, temp=0.05):
        super().__init__()
        self.fc = nn.Linear(inc, num_class, bias=False)
        self.num_class = num_class
        self.temp = temp

    def forward(self, x, reverse=False, eta=0.1):
        if reverse:
            x = grad_reverse(x, eta)
        
        x_out = self.fc(x) / self.temp
        return x_out
    
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn_heads = 8
        
        self.net = nn.Sequential(*list(
            r2plus1d_34_32_ig65m(359,pretrained = True,progress = True).children())[:-1])
            #r2plus1d_34_32_kinetics(400,pretrained = True,progress = True).children())[:-1])
        #self.bert = BERT5(512, 1 , hidden=512, n_layers=3, attn_heads=8)
        

    def forward(self,input):
        output = self.net(input)

        #output , maskSample = self.bert(output.view(-1,1,512))  
        #output = output[:,0,:]  
        
        output = F.normalize(output.view(-1,512),p=2 )
        
        
        return output
    

