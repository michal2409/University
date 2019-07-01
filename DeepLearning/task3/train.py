import model.net as net
import model.data_loader as data_loader
from preproc import *
from util import *
from torch import FloatTensor
from tqdm import tqdm

torch.manual_seed(0)
text = open("data/train.txt", "r").read()
sent = text.split('\n')
corp = CorpusPreprocessor(text)
train_loader = data_loader.fetch_dataloader(sent, corp)

model = net.net(n_letters, 256, 3, 2, 100)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

for epoch in range(10):
    losss = 0.0
    correct, total = 0, 0
    for i, (x, z, y) in enumerate(tqdm(train_loader)):
        x, y, z = sentToTensor(x[0]).to(device), y.to(device), lineToTensor(z[0]).to(device)
        optimizer.zero_grad()
        outputs = model(x, z)

        loss = criterion(outputs, y)
        losss += loss
        loss.backward()
        optimizer.step()  
        
        _, predicted = torch.max(outputs.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

torch.save(model.state_dict(), "model.pth")