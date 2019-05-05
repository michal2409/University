import model.net as net
import model.data_loader as data_loader
import numpy as np
import torch.nn as nn
import torch

if __name__ == '__main__':
	test_loader = data_loader.fetch_dataloader('/scidata/fruits-360/Test', batch_size=128)
	device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
	model = net.Net()
	model.load_state_dict(torch.load("model.pth"))
	model.to(device)
	model.eval()

	correct, total = 0, 0
	with torch.no_grad():
		for images, labels in test_loader:
			images, labels = images.to(device), labels.to(device)

			outputs = model(images)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()

	print('Accuracy on test data: {:.4f} %'.format(100 * correct / total))



