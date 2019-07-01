import model.net as net
import model.data_loader as data_loader
from preproc import *
from util import *
import random
from tqdm import tqdm

torch.manual_seed(0)

text = open("data/test.txt", "r").read()
sent = text.split('\n')

corp = CorpusPreprocessor(text)
test_loader = data_loader.fetch_dataloader(sent, corp)

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
model = net.net(n_letters, 256, 3, 2, 100)
model.load_state_dict(torch.load("model.pth"))
model.eval()
model.to(device)

correct, total = 0, 0
with torch.no_grad():
    for i, (x, z, y) in enumerate(tqdm(test_loader)):
        x, y, z = lineToTensor(x[0]).to(device), y.to(device), lineToTensor(z[0]).to(device)
        outputs = model(x, z)
        
        _, predicted = torch.max(outputs.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()
        
print('Accuracy on test data: {:.4f} %'.format(100 * correct / total))