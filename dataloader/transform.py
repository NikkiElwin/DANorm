import torch

import numpy as np
import random as ra
import csv
import os
import cv2





class NormalizeLen(object):
    """ Return a normalized number of frames. """

    def __init__(self, vid_len=24):
        self.vid_len = vid_len

    def __call__(self, sample):
        video,label = sample['video'],sample['label']
        num_frames_rgb = len(video)
        indices_rgb = np.linspace(0, num_frames_rgb - 1, self.vid_len).astype(int)
        video = video[indices_rgb, :, :, :]


        return {
            'video': video,
            'label': label
        }

class AugCrop(object):
    """ Return a temporal crop of given sequences """

    def __init__(self, p_interval=0.5):
        self.p_interval = p_interval

    def __call__(self, sample):
        video,label = sample['video'],sample['label']
        ratio = (1.0 - self.p_interval * np.random.rand())
        num_frames_rgb = len(video)
        begin_rgb = (num_frames_rgb - int(num_frames_rgb * ratio)) // 2
        video = video[begin_rgb:(num_frames_rgb - begin_rgb), :, :, :]


        return {
            'video': video,
            'label': label
        }

class ToTensor(object):
    def __call__(self,sample):
        video,label = sample['video'],sample['label']
        return {
            'video': torch.from_numpy(video.astype(np.float32)),
            'label': torch.from_numpy(np.asarray(label)).long()
        }
class Reshape(object):
    def __call__(self,sample):
        video,label = sample['video'],sample['label']
        video = video.transpose(3,0,1,2)
        
        return {
            'video': video,
            'label': label
        }
        
