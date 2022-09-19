from enum import Flag
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import time
from typing import Callable
import pandas as pd
import numpy as np
import random as ra
import csv
import os
import cv2
from tqdm import tqdm
import argparse


from models.network import Net,Classifier
from dataloader.dataset import RawDataset,PreprocessDataset
from dataloader.transform import AugCrop,Reshape,ToTensor,NormalizeLen
from utils.utils import accuracy,AverageMeter


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=int, default=1, metavar='N')           
parser.add_argument('--seed', type=int, default=42, metavar='N')       
args = parser.parse_args()


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
seed = args.seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  
np.random.seed(seed)  # Numpy module.
ra.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices: a list of indices
        num_samples: number of samples to draw
        callback_get_label: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(self, dataset, indices: list = None, num_samples: int = None, callback_get_label: Callable = None):
        # if indices is not provided, all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided, draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        # distribution of classes in the dataset
        df = pd.DataFrame()
        df["label"] = self._get_labels(dataset)
        df.index = self.indices
        df = df.sort_index()

        label_to_count = df["label"].value_counts()

        weights =  1 / (label_to_count[df["label"]])
        
        self.weights = torch.DoubleTensor(weights.to_list())

    def _get_labels(self, dataset):
        if self.callback_get_label:
            return self.callback_get_label(dataset)
        elif isinstance(dataset, torchvision.datasets.MNIST):
            return dataset.train_labels.tolist()
        elif isinstance(dataset, torchvision.datasets.ImageFolder):
            return [x[1] for x in dataset.imgs]
        elif isinstance(dataset, torchvision.datasets.DatasetFolder):
            return dataset.samples[:][1]
        elif isinstance(dataset, torch.utils.data.Subset):
            return dataset.dataset.imgs[:][1]
        elif isinstance(dataset, torch.utils.data.Dataset):
            return dataset.get_labels()
        else:
            raise NotImplementedError

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples

class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    
    Reference:
        Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    
    Imported from `<https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py>`_.
    
    Args:
        margin (float, optional): margin for triplet. Default is 0.3.
    """
    
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
 
    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (num_classes).
        """
        n = inputs.size(0)	# batch_size
        
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(inputs, inputs.t(),beta = 1,alpha= -2)
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss


class RingLoss(nn.Module):
    def __init__(self, type='auto', loss_weight=1.0):
        """
        :param type: type of loss ('l1', 'l2', 'auto')
        :param loss_weight: weight of loss, for 'l1' and 'l2', try with 0.01. For 'auto', try with 1.0.
        :return:
        """
        super(RingLoss, self).__init__()
        self.radius = nn.parameter.Parameter(torch.Tensor(1)).cuda()
        self.radius.data.fill_(-1)
        self.loss_weight = loss_weight
        self.type = type

    def forward(self, x):
        x = x.pow(2).sum(dim=1).pow(0.5)
        if self.radius.data[0] < 0: # Initialize the radius with the mean feature norm of first iteration
            self.radius.data.fill_(x.mean().data)
        if self.type == 'l1': # Smooth L1 Loss
            loss1 = F.smooth_l1_loss(x, self.radius.expand_as(x)).mul_(self.loss_weight)
            loss2 = F.smooth_l1_loss(self.radius.expand_as(x), x).mul_(self.loss_weight)
            ringloss = loss1 + loss2
        elif self.type == 'auto': # Divide the L2 Loss by the feature's own norm
            diff = x.sub(self.radius.expand_as(x)) / (x.mean().detach().clamp(min=0.5))
            diff_sq = torch.pow(torch.abs(diff), 2).mean()
            ringloss = diff_sq.mul_(self.loss_weight)
        else: # L2 Loss, if not specified
            diff = x.sub(self.radius.expand_as(x))
            diff_sq = torch.pow(torch.abs(diff), 2).mean()
            ringloss = diff_sq.mul_(self.loss_weight)
        return ringloss


class GradualWarmupScheduler(optim.lr_scheduler._LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """
 
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)
 
    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
 
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
 
    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)
 
    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != optim.lr_scheduler.ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)



if __name__ == "__main__":
    index2labels = ['drink', 'jump', 'pick', 'pour', 'push', 'run', 'sit', 'stand', 'turn', 'walk', 'wave']
    batch = 4
    num_workers = 2
    imgSize = (112,112)
    path = './dataset/'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainTransform = transforms.Compose([AugCrop(0.6),NormalizeLen(48),Reshape(),ToTensor()])
    testTransform = transforms.Compose([NormalizeLen(48),Reshape(),ToTensor()])
    dataset = RawDataset(path,index=args.dataset)
    EPOCH = 300
    
    
    best = 0.0
    maxStep = 100
    net = Net().to(device)
    classifier = Classifier().to(device)
    optimizerE = optim.AdamW(net.parameters(),lr = 1e-5,weight_decay=0.01)
    optimizerC = optim.AdamW(classifier.parameters(),lr = 1e-5,weight_decay=0.01)
    

    
    criteria = nn.CrossEntropyLoss()
    ringLoss = RingLoss()
    tpLoss = TripletLoss()
    
    th = 0.8
    newSample = 0
    losses = AverageMeter()
    for epoch in range(EPOCH):
        labelSet = PreprocessDataset(dataset.labelSet,imgSize = imgSize,transform = trainTransform)
        unlabelSet = PreprocessDataset(dataset.unlabelSet,imgSize = imgSize,transform = testTransform,isDark = True)
        darkSet = PreprocessDataset(dataset.darkSet,imgSize = imgSize,transform = testTransform,length=batch * maxStep)
        
        trainSampler = ImbalancedDatasetSampler(labelSet)
        trainData = DataLoader(labelSet,batch_size = batch,num_workers = num_workers,sampler=trainSampler)
        testData = DataLoader(unlabelSet,batch_size = 1,num_workers = num_workers)
        darkData = DataLoader(darkSet,batch_size = 1,num_workers = num_workers,shuffle = True)

        print("Lr:", optimizerE.param_groups[0]['lr'],optimizerC.param_groups[0]['lr'])
        net.train(True)
        processBar = tqdm(enumerate(trainData,1),total=len(trainData))
        
        acc = [AverageMeter(),AverageMeter()]
        #darkIter = iter(darkData)
        for step,sample in processBar:
            
            sample['video'] = sample['video'].to(device)
            sample['label'] = sample['label'].to(device)
            sample['weight'] = sample['weight'].float().to(device)
            net.zero_grad()
            classifier.zero_grad()
            feature = net(sample['video'])
            
            pred = classifier(feature)
            
            loss = F.nll_loss(F.log_softmax(pred,1),sample['label'],reduction='none')
            clsLoss = torch.mean(loss * sample['weight'])
            rLoss = ringLoss(pred)
            
            loss = clsLoss + rLoss
            losses.update(loss.item())

            prec1, prec5 = accuracy(pred, sample['label'], topk=(1, 5))
            acc[0].update(prec1)
            acc[1].update(prec5)

            
            loss.backward()
            optimizerE.step()
            optimizerC.step()
            processBar.set_description("[%d] Acc@1: %.2f Acc@5: %.2f Loss: %.8f ( c %.4f | r %.4f )" % 
                                       (epoch,acc[0].avg,acc[1].avg,losses.avg,clsLoss.item(),rLoss.item()))
            if step > maxStep: break
            
        with torch.no_grad():
            net.train(False)
            processBar = tqdm(enumerate(testData,1),total=len(testData))
            losses = AverageMeter()
            results = list()
            acc = [AverageMeter(),AverageMeter()]
            totalAcc,total = 0,0
            for step,sample in processBar:
                
                sample['video'] = sample['video'].to(device)
                sample['label'] = sample['label'].to(device)
                feature = net(sample['video'])
                pred = classifier(feature)
                pred = torch.softmax(pred,dim = -1)
                
                score,index = torch.max(pred,dim = -1)
                
                prec1, prec5 = accuracy(pred, sample['label'], topk=(1, 5))
                acc[0].update(prec1)
                acc[1].update(prec5)
                processBar.set_description("[Valid] Acc@1: %.2f Acc@5: %.2f" % (acc[0].avg,acc[1].avg))
            headers = ['VideoID','Video','ClassID','Probability']
            
            if acc[0].avg > 85:
                state_dict = {'encoder': net.state_dict(),'classifier': classifier.state_dict()}
                torch.save(state_dict,'./checkpoint/DANorm-%d-%.2f.pth ' % (epoch,acc[0].avg))
            if acc[0].avg > best:
                print("==========================Best========================")
                print("Acc: %.4f" % (acc[0].avg))
                

                best = acc[0].avg
            
            net.train(False)
            processBar = tqdm(enumerate(darkData,1),total=len(darkData))
            losses = AverageMeter()
            results = list()
            acc = [AverageMeter(),AverageMeter()]
            for step,sample in processBar:
                
                sample['video'] = sample['video'].to(device)

                feature = net(sample['video'])
                pred = classifier(feature)
                pred = torch.softmax(pred,dim = -1)
                
                score,index = torch.max(pred,dim = -1)

                results.append({
                    'VideoID': sample['id'].item(),
                    'Probability': pred.cpu().detach(),
                    'ClassID': int(index),
                    'Video': os.path.split(sample['path'][0])[-1]
                })

                processBar.set_description("[Valid] " )
            
            
        if(dataset.update(results,epoch = epoch/EPOCH,th = th,isValid = False)):
            th += 0.01
        if th >= 0.99:
            th = 0.99
        
        
        print("")
        torch.cuda.empty_cache()
    filename = 'Report.txt'
    with open(filename,'a') as file:
        file.write("\n********************************************")
        file.write("\nTime: " + time.asctime(time.localtime(time.time())) + "\n")
        file.write("Dataset: %d\n" % (args.dataset))
        file.write("best" +": %.2f" % (best) + "\n")
        file.write("seed: %d" % (seed) + "\n")
        file.write("********************************************\n")