import model.net as net
import model.data_loader as data_loader
import numpy as np
import torch.nn as nn
import torch
import datetime
import time

from util import load_model
from torchvision import transforms

def test(test_loader, model = None):
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    if (model == None):
        model = load_model(device)
    
    correct, total = 0, 0
    with torch.no_grad():
        for images, _, labels, _ in test_loader:
            flipped_img = torch.flip(images, [3]).to(device)  # horizontal flip of image
            images, labels = images.to(device), labels.to(device).type(torch.cuda.LongTensor)
            
            outputs = model(images)
            outputs_flipped = torch.flip(model(flipped_img), [3])
            avg_output = (outputs + outputs_flipped) / 2
            _, predicted = torch.max(avg_output.data, 1)
            
            total += labels.shape[0] * labels.shape[1] * labels.shape[2]
            correct += (predicted == labels).sum().item()
            
    return correct, total

if __name__ == '__main__':
    test_loader = data_loader.fetch_dataloader('data/val', batch_size=8)
    correct, total = test(test_loader)
    t = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    print('Time : {} Accuracy on test data: {:.4f} %'.format(t, 100 * correct / total))
