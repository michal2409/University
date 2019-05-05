from torch.autograd import Variable
import numpy as np
import torch
import utils as ut
import random
import sys
    
if __name__ == '__main__':
    '''Command line args:
        argv[1]: partial directory to image(fruit_name/img.jpg)
        argv[2]: label of image
        argv[3]: size of square used for occlusion
    '''
    assert (len(sys.argv) == 4)
    img_path, label, occl_size = '/scidata/fruits-360/Test/' + str(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model = ut.load_model(device)
    img = ut.img_to_tensor(img_path)
    
    '''For each pixel in the input image making occlusion with
        square centered in that pixel filled with zeros.
        For such new image making forward pass and saving 
        softmax score in corresponding place in heatmap.
    '''
    _, _, h, w = img.shape
    heatmap = torch.zeros(h, w)
    r = occl_size // 2

    for y in range(h):
        for x in range(w):
            h_start = max(0, y - r)
            h_end = min(h, y + r)
            w_start = max(0, x - r)
            w_end = min(w, x + r)
            
            input_image = torch.from_numpy(np.copy(img.data))
            input_image[:, :, h_start:h_end, w_start:w_end] = 0
            out = model(Variable(input_image.to(device)))
            softmax = torch.nn.functional.softmax(out, dim = 1).data[0]
            heatmap[x][y] = softmax[label]
            
    ut.save_plot(img, heatmap, 'Occlusion' + str(random.randint(1,101)) + '.jpg')
    