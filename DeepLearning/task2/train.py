import model.net as net
import model.data_loader as data_loader
import numpy as np
import torch.nn as nn
import torch
import datetime
import time
import sys

from test import test
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from util import RandomHorizontalFlip, check_array

if __name__ == '__main__':
    train_loader = data_loader.fetch_dataloader('data/train', batch_size=8, augment=RandomHorizontalFlip(0.5))
    test_loader = data_loader.fetch_dataloader('data/val', batch_size=8)
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    model = net.Unet(check_array.shape[0]).to(device)
    model.train()
    
    writer = SummaryWriter()
    global_step = 0
    step = 0

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    num_epochs = 10

    for epoch in range(num_epochs):
        loss_epoch, correct, total = 0.0, 0, 0
        scheduler.step()
        for i, (images, _, labels, _) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device).type(torch.cuda.LongTensor)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_epoch += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total_batch = labels.shape[0] * labels.shape[1] * labels.shape[2]
            correct_batch = (predicted == labels).sum().item()
            total += total_batch
            correct += correct_batch
            
            if step % 100 == 0:
                writer.add_scalar("train batch loss", loss.item(), global_step=global_step)
                writer.add_scalar("train batch accuracy", 100*correct_batch/total_batch, global_step=global_step)
                global_step += 1
                
            step += 1
            
        writer.add_scalar("test accuracy", 100*correct_batch/total_batch, global_step=epoch)
        model.eval()
        test_correct, test_total = test(test_loader, model)
        model.train()
        t = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
        print ('Time: {} Epoch [{}/{}] Train Acc: {:.4f} Test Acc: {:.4f}'.format(t, epoch+1, num_epochs, (100 * correct / total), (100 * test_correct / test_total)))

    torch.save(model.state_dict(), "model.pth")
    writer.close()
