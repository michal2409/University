import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from batch_norm import Batch_norm

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(3, 16, kernel_size=5)
		self.conv1_bn = Batch_norm(16, training=self.training)
		self.pool = nn.MaxPool2d(2, 2)

		self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
		self.conv2_bn = Batch_norm(32, training=self.training)

		self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
		self.conv3_bn = Batch_norm(64, training=self.training)

		self.fc1 = nn.Linear(64 * 10 * 10, 250)
		self.fc1_bn = Batch_norm(250, training=self.training)
		self.fc2 = nn.Linear(250, 95)

	def forward(self, x):
		x = self.pool(F.relu(self.conv1_bn(self.conv1(x))))
		x = self.pool(F.relu(self.conv2_bn(self.conv2(x))))
		x = self.pool(F.relu(self.conv3_bn(self.conv3(x))))

		x = x.view(-1, 64 * 10 * 10)
		x = F.dropout(F.relu(self.fc1_bn(self.fc1(x))), training=self.training, p=0.5)
		x = self.fc2(x)

		return F.log_softmax(x, dim=1)