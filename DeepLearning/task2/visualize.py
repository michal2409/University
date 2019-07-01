import model.data_loader as data_loader
import numpy as np
import imageio
import torch
import os

from tqdm import tqdm
from torch.autograd import Variable
from torchvision import transforms
from datetime import datetime
from matplotlib import pyplot as plt
from numba import uint16, jit
from util import create_dir_if_not_exist, load_model, check_array

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
transformer = transforms.Compose([transforms.ToTensor()])

def plot(preds, y):
    image = np.zeros((256, 2*256, 3), dtype=np.uint16)
    for r in range(256):
        for c in range(256):
            image[r][c] = check_array[preds[r][c]]
    image[:, 256:, :] = y
    return image
    
plot_fast = jit(uint16[:,:](uint16[:,:], uint16[:,:,:]))(plot)

def plot_image(model, path, path_y = None):
    """Plots prediction for given image.
       If path_y=None then is assumed image
       is from cityscapes directory i.e originial
       image and segmented are joined.
    """
    img = plt.imread(path) * 255
    if path_y:
        X, y = img, plt.imread(path_y) * 255
    else:
        X, y = img[:, :256, :], img[:, 256:, :]
    
    with torch.no_grad():
        X, y = transformer(X), torch.from_numpy(y)
        X.unsqueeze_(0)
        y.unsqueeze_(0)
        X = Variable(X.to(device))

        outputs = model(X)
        _, preds = torch.max(outputs.data, 1)
        fname = os.path.basename(os.path.normpath(path))
        preds, y = preds.cpu().numpy(), y.numpy()

        image = plot_fast(preds, y)
        imageio.imwrite('preds/' + fname, image.astype(np.uint8))

def plot_validation(model):
    """Plots predictions for whole validation set"""
    test_loader = data_loader.fetch_dataloader('data/val', batch_size=8)
    with torch.no_grad():
        for images, y, _, fname in tqdm(test_loader):
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs.data, 1)
            preds, y = preds.cpu().numpy(), y.numpy()
            for b in range(preds.shape[0]):
                image = plot_fast(preds[b], y[b])
                imageio.imwrite('preds/' + fname[b], image.astype(np.uint8))               

if __name__ == '__main__':
    model = load_model(device)
    create_dir_if_not_exist('preds')
    plot_validation(model)
