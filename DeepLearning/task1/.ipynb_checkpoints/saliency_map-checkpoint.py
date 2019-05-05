from torch.autograd import Variable
import model.net as net
import numpy as np
import torch
import utils as ut
import random
import sys
    
if __name__ == '__main__':
    '''Command line args:
        argv[1]: partial directory to image(fruit_name/img.jpg)
        argv[2]: label of image
    '''
    assert(len(sys.argv) == 3)
    img_path, label = '/scidata/fruits-360/Test/' + str(sys.argv[1]), int(sys.argv[2])
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model = ut.load_model(device)
    img = ut.img_to_tensor(img_path)
    
    # Creating heatmap from pixelwise gradient of loss function.
    image_var = Variable(img.to(device), requires_grad=True)
    scores = model(image_var)
    scores[0][label].backward()
    
    saliency = image_var.grad.data
    saliency = saliency.abs()
    saliency.squeeze_()
    saliency.transpose_(0,1)
    saliency.transpose_(1,2)
    saliency = np.max(saliency.cpu().numpy(), axis=2)
    
    ut.save_plot(img, saliency, 'Saliency' + str(random.randint(1,101)) + '.jpg')
    