import os
import random
import torch
import numpy as np
import model.net as net

check_array = np.array(
    [[116, 17, 36],
     [152, 43,150],
     [106,141, 34],
     [ 69, 69, 69],
     [  2,  1,  3],
     [127, 63,126],
     [222, 52,211],
     [  2,  1,140],
     [ 93,117,119],
     [180,228,182],
     [213,202, 43],
     [ 79,  2, 80],
     [188,151,155],
     [  9,  5, 91],
     [106, 75, 13],
     [215, 20, 53],
     [110,134, 62],
     [  8, 68, 98],
     [244,171,170],
     [171, 43, 74],
     [104, 96,155],
     [ 72,130,177],
     [242, 35,231],
     [147,149,149],
     [ 35, 25, 34],
     [155,247,151],
     [ 85, 68, 99],
     [ 71, 81, 43],
     [195, 64,182],
     [146,133, 92]],
     dtype = np.uint8)

def load_model(device):
    model = net.Unet(check_array.shape[0])
    model.load_state_dict(torch.load("model.pth"))
    model.to(device)
    model.eval()
    return model

def create_dir_if_not_exist(path):
    if not os.path.exists(path):
        os.mkdir(path)
        
class RandomHorizontalFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            img, mask = np.fliplr(img).copy(), np.fliplr(mask).copy()
        
        return img, mask