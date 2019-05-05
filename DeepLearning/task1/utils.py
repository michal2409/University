from PIL import Image
import torchvision
import model.net as net
import matplotlib.pyplot as plt
import torch

def img_to_tensor(path):
    '''Transforms PIL image to tensor of shape [1, C, H, W]
    
    Args:
        path: path to image
    '''
    jpgfile = Image.open(path)
    pil2tensor = torchvision.transforms.ToTensor()
    img_tensor = pil2tensor(jpgfile)
    img_tensor.unsqueeze_(0)
    return img_tensor

def load_model(device):
    '''Loads model and sets into testing phase
    
    Args:
        device: device for model
    '''
    model = net.Net()
    model.load_state_dict(torch.load("model.pth"))
    model.to(device)
    model.eval()
    return model

def save_plot(img1, img2, title):
    '''Saves plot of side by side img1 and img2 into img folder
    
    Args:
        img1: first image(tensor [1, C, H, W])
        img2: second image(numpy [H, W])
        title: title of saving image
    '''
    fig = plt.figure()
    plt.rcParams["figure.figsize"] = (40, 40)

    a = fig.add_subplot(1,2,1)
    imgplot = plt.imshow(img1.squeeze().permute(1, 2, 0))
    plt.axis('off') 

    a = fig.add_subplot(1,2,2)
    imgplot = plt.imshow(img2)
    plt.axis('off') 
    
    plt.savefig('img/' + str(title))