import model.net as net
import model.data_loader as data_loader
import numpy as np
import torch.nn as nn
import torch
import datetime
import time
import sys

if __name__ == '__main__':
	train_loader = data_loader.fetch_dataloader('/scidata/fruits-360/Training', batch_size=128)
	device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
	model = net.Net().to(device)
	model.train()

	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
	num_epochs = 5

	for epoch in range(num_epochs):
		loss_epoch, correct, total = 0.0, 0, 0
		scheduler.step()
		for i, (images, labels) in enumerate(train_loader):
			images, labels = images.to(device), labels.to(device)

			optimizer.zero_grad()

			outputs = model(images)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			loss_epoch += loss.item()
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()

		t = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
		print ('Time: {} Epoch [{}/{}] Acc: {:.4f}'.format(t, epoch+1, num_epochs, (100 * correct / total)))

	torch.save(model.state_dict(), "model.pth")