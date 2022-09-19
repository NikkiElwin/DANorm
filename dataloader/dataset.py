from tkinter.messagebox import NO
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset,SubsetRandomSampler
import torchvision
import torchvision.transforms as transforms

import numpy as np
import random as ra
import csv
import os
import cv2

#index2labels = ['Drink', 'Jump', 'Pick', 'Pour', 'Push', 'Run', 'Sit', 'Stand', 'Turn', 'Walk', 'Wave']

def gamma_intensity_correction(img, gamma):
    """
    :param img: the img of input
    :param gamma:
    :return: a new img
    """
    invGamma = 1.0 / gamma
    LU_table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    gamma_img = cv2.LUT(img, LU_table)
    return gamma_img



def loadVideo(path, imgSize = (256,256),vidLen=24,isDark = False):
    cap = cv2.VideoCapture(path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    
    height,width = imgSize

    # Init the numpy array
    video = np.zeros((vidLen,  height, width, 3)).astype(np.float32)
    taken = np.linspace(0, num_frames, vidLen).astype(int)

    np_idx = 0
    for fr_idx in range(num_frames):
        ret, frame = cap.read()
        if cap.isOpened() and fr_idx in taken:
            if isDark:
                pass
                #frame = gamma_intensity_correction(frame,1.2)
            frame = cv2.resize(frame,(height,width))

            video[np_idx, :, :, :] = frame.astype(np.float32)
            np_idx += 1

    cap.release()

    return video

        
class RawDataset(Dataset):
    def __init__(self,path,file = 'avi',index = 1):
        super().__init__()
        self.labelPath = os.path.join(path,'ug2_2022_train_labeled.csv')
        self.videoPath = os.path.join(path,'Train')
        self.unlabelPath = os.path.join(path,'ug2_2022_test_labeled.csv')#os.path.join(path,'ARID','list_cvt','AID11_split%d_test.txt' % (index))
        self.unlabelVideoPath = '../Zero_DCE/Test-light'#os.path.join(path,'Test')#os.path.join(path,'ARID',file)#
        
        self.darkVideoPath = '../Zero_DCE/results'#os.path.join(path,'dark-train')#
        self.darkPath = os.path.join(path,'ug2_2022_train_dark.csv')
            
        self.unlabelSet = {}
        
        with open(self.unlabelPath,'r') as file:
            next(file) #Remove the first line
            reader = csv.reader(file)    
            for item in reader:
                label,path = item[2].split() if item[1] == '' else (item[2],item[1])
                _path = os.path.join(self.unlabelVideoPath,path)[:-3] + 'avi'
                _id = int(item[0]) 
                assert os.path.exists(_path), "File %s is not exist!" % _path
                self.unlabelSet[_id] = {
                        'path': _path,
                        'class': int(label),
                        'weight': 1.0,
                        'id': _id
                    }
            
        self.darkSet = {}
        self.darkLen = len(self.unlabelSet.keys())
        with open(self.darkPath,'r') as file:
            next(file) #Remove the first line
            reader = csv.reader(file)    
            for item in reader:
                _path = os.path.join(self.darkVideoPath,item[1])[:-3] + 'avi'
                _id = int(item[0])+ self.darkLen
                assert os.path.exists(_path), "File %s is not exist!" % _path
                self.darkSet[_id] = {
                        'path': _path,
                        'class': -1,
                        'weight': 1.0,
                        'id': _id
                    }
                
        
                
        self.labelSet = {}
        self.unlabelLen = len(self.unlabelSet.keys()) + len(self.darkSet.keys())
        with open(self.labelPath,'r') as file:
            next(file) #Remove the first line
            reader = csv.reader(file)    
            for item in reader:
                label,path = item[2].split() if item[1] == '' else (item[2],item[1])
                _path = os.path.join(self.videoPath,path)
                _id = int(item[0]) + self.unlabelLen
                assert os.path.exists(_path), "File %s is not exist!" % _path
                self.labelSet[_id] = {
                        'path': _path,
                        'class': int(label),
                        'weight': 1.0,
                        'id': _id
                    }
                
        
        
    def update(self,results,epoch,isValid = True,th = 0.9):
        _length = len(self.labelSet.keys())
        _update = 0
        for result in results:
            id,pred,path = result['VideoID'],result['Probability'],result['Video']
            score,index = torch.max(pred,dim = -1)
            
            
            if score.item() > th:
                id = id + self.darkLen if isValid else id
                if id in self.labelSet.keys():
                    if self.labelSet[id]['class'] != index.item():
                        _update += 1
                        
                _path = os.path.join(self.unlabelVideoPath,path) if isValid else os.path.join(self.darkVideoPath,path)
                self.labelSet[id] = {
                        'path': _path,
                        'class': index.item(),
                        'weight': epoch * score.item(),
                        'id': id
                    }
                
                #del self.unlabelSet[int(id)]
        print("New sample: %d, Update sample: %d" % (len(self.labelSet.keys()) - _length,_update))
        return len(self.labelSet.keys()) - _length
        
class PreprocessDataset(Dataset):
    def __init__(self,dataset,transform = None,length = None,isDark = False,imgSize = None):
        super().__init__()
        self.dataset = list(dataset.values())
        self.length = length
        self.isDark = isDark
        self.imgSize = imgSize
        
        ra.shuffle(self.dataset)
        self.transform = transform
        if length is None:
            self.labels = [data['class'] for data in self.dataset]
        else:
            self.labels = [data['class'] for data in self.dataset][:self.length]
        
    def get_labels(self):
        return self.labels
        
    def __len__(self):
        if self.length is None:
            return len(self.dataset)
        else:
            return self.length
    
    def __getitem__(self,index):
        data = self.dataset[index]
        video = np.zeros([1])

        video = loadVideo(data['path'],self.imgSize,vidLen = 64,isDark=self.isDark)
        video = video/255.0
        video -= np.asarray([0.485, 0.456, 0.406]).reshape(1, 1, 3)  # mean
        video /= np.asarray([0.229, 0.224, 0.225]).reshape(1, 1, 3)  # std
        
        sample = {
            'video': video,
            'label': data['class']
        }
        if self.transform:
            sample = self.transform(sample)
        sample['id'] = data['id']
        sample['path'] = data['path']
        sample['weight'] = torch.tensor(data['weight']).float()
        return sample

    