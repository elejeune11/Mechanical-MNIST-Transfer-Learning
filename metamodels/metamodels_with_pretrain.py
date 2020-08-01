from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from torch.utils.data import Dataset, DataLoader
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from PIL import Image
import numpy as np
import os
##########################################################################################
# info to set 
##########################################################################################
data_retrain = 'UE'
# num_retrain = [100,1000,10000] # from paper 
num_retrain = [100] 
# model_pretrain = ['UE_CM_28_perturb_num_1000', 'UE_CM_28_perturb_num_10000', 'UE_CM_28_perturb_num_60000'] # from paper 
model_pretrain = ['UE_CM_28_perturb_num_1000'] 

for NR in num_retrain:
	fname = data_retrain
	mini_size_train = NR
	mini_size_test = 10000

	in_train_fname   = 'NPY_ARRAYS/' + fname + '_num_' + str(mini_size_train) + '_MNIST_bitmap_train.npy'
	in_test_fname    = 'NPY_ARRAYS/' + fname + '_num_' + '10000' + '_MNIST_bitmap_test.npy'
	out_train_fname  = 'NPY_ARRAYS/' + fname + '_num_' + str(mini_size_train) + '_final_psi_train.npy'
	out_test_fname   = 'NPY_ARRAYS/' + fname + '_num_' + '10000' + '_final_psi_test.npy'

	for PTM in model_pretrain:
		MAE_test_all = [] 
		MAE_train_all = [] 
		is_pretrained = True
		pretrained_model_name = PTM
		model_options = 1
		print('STARTING:',data_retrain, NR, 'pretrained on:',PTM)
		##########################################################################################
		# import data
		##########################################################################################
		class MechMNISTDataset(Dataset):
			""" mechanical MNIST data set"""
 
			def __init__(self,train=True,transform=None, target_transform=None):
				self.train = train
				if self.train:
					self.data = np.load(in_train_fname)
					self.data = self.data.reshape((self.data.shape[0],28,28))
					self.targets = np.load(out_train_fname).reshape(-1,1).astype(float)
				else:
					self.data = np.load(in_test_fname)
					self.data = self.data.reshape((self.data.shape[0],28,28))
					self.targets = np.load(out_test_fname).reshape(-1,1).astype(float)
				self.transform = transform
				self.target_transform = target_transform
	
			def __len__(self):
				return self.data.shape[0]

			def __getitem__(self,idx):
				if torch.is_tensor(idx):
					idx = idx.tolist()
	
				img = self.data[idx,:,:]
				lab = self.targets[idx]
	
				#img = Image.fromarray(img.numpy(), mode='L')

				if self.transform is not None:
					img = self.transform(img)

				if self.target_transform is not None:
					lab = self.target_transform(lab)
	
				sample = (img,lab)
	
				return sample 
		##########################################################################################
		# model
		##########################################################################################
		class Net(nn.Module): 
			def __init__(self):
				super(Net, self).__init__()
				self.conv1 = nn.Conv2d(1, 20, 5, 1)
				self.conv2 = nn.Conv2d(20, 50, 5, 1)
				self.fc1 = nn.Linear(4*4*50, 50)
				self.fc2 = nn.Linear(50, 1) 

			def forward(self, x):
				x = F.relu(self.conv1(x))
				x = F.max_pool2d(x, 2, 2)
				x = F.relu(self.conv2(x))
				x = F.max_pool2d(x, 2, 2)
				x = x.view(-1, 4*4*50)
				x = F.relu(self.fc1(x))
				x = self.fc2(x)
				return x 

		##########################################################################################
		def train(args, model, device, train_loader, optimizer, epoch):
			model.train()
			loss_fcn = nn.MSELoss()
			# CHANGED loss_fcn = nn.L1Loss()
			for batch_idx, (data, target) in enumerate(train_loader):
				data, target = data.to(device), target.to(device)
				optimizer.zero_grad()
				output = model(data)
				loss = loss_fcn(output,target.float()) #REGcha
				loss.backward()
				optimizer.step()
				if batch_idx % args.log_interval == 0:
					print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
						epoch, batch_idx * len(data), len(train_loader.dataset),
						100. * batch_idx / len(train_loader), loss.item()))

		##########################################################################################
		def test(args, model, device, test_loader):
			model.eval()
			loss_fcn = nn.MSELoss()
			#loss_fcn = nn.SmoothL1Loss()
			# CHANGED loss_fcn = nn.L1Loss()
			test_loss = 0
			correct = 0
			MAE = 0 
			counter = 0
			with torch.no_grad():
				for data, target in test_loader:
					data, target = data.to(device), target.to(device)
					output = model(data)
					test_loss += loss_fcn(output, target.float()).item() # sum up batch loss
					pred = output
					counter += 1 
					MAE += np.abs(target.detach().numpy()[0][0] - output.detach().numpy()[0][0])

			test_loss /= len(test_loader.dataset)

			MAE = MAE/counter

			print('\nTest set: Average loss: {:.4f}, MAE: {:.4f} \n'.format(test_loss, MAE))

			return MAE

		##########################################################################################
		# Training settings
		parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
		#batch size = 64
		parser.add_argument('--batch-size', type=int, default=1, metavar='N', help='input batch size for training (default: 1)')
		parser.add_argument('--test-batch-size', type=int, default=10, metavar='N',
							help='input batch size for testing (default: 1000)')
		parser.add_argument('--epochs', type=int, default=20, metavar='N', help='number of epochs to train (default: 20)')
		parser.add_argument('--lr', type=float, default=0.001, metavar='LR',help='learning rate (default: 0.001)')
		parser.add_argument('--momentum', type=float, default=0.1, metavar='M',
							help='SGD momentum (default: 0.5)')
		parser.add_argument('--no-cuda', action='store_true', default=False,
							help='disables CUDA training')
		parser.add_argument('--seed', type=int, default=2, metavar='S',
							help='random seed (default: 1 then 2)')
		parser.add_argument('--log-interval', type=int, default=10, metavar='N',
							help='how many batches to wait before logging training status')
		parser.add_argument('--save-model', action='store_true', default=True,
							help='For Saving the current Model')
		args = parser.parse_args()
		use_cuda = not args.no_cuda and torch.cuda.is_available()
		torch.manual_seed(args.seed)
		device = torch.device("cuda" if use_cuda else "cpu")
		kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
		train_loader = DataLoader( MechMNISTDataset(train=True,  transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])), batch_size=args.batch_size, shuffle=True, **kwargs)
		test_loader  = DataLoader( MechMNISTDataset(train=False, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])), batch_size=args.batch_size, shuffle=True, **kwargs)

		model = Net().to(device)
		if is_pretrained:
			model.load_state_dict(torch.load(pretrained_model_name + '.pt', map_location='cpu'))
		optimizer = optim.Adam(model.parameters(), lr=args.lr)
		
		# save @ every epoch -- ultimately choose the epoch with the lowest training error as the final pre-trained model
		folder = fname + '_num_' + str(mini_size_train) + '_pretrained_with_' + pretrained_model_name
		if not os.path.exists(folder):
			os.mkdir(folder)
			
		for epoch in range(1, args.epochs + 1):
			train(args, model, device, train_loader, optimizer, epoch)
			MAE_test = test(args, model, device, test_loader) # test error
			MAE_train = test(args, model, device, train_loader) # training error 
			MAE_test_all.append(MAE_test)
			MAE_train_all.append(MAE_train)
			torch.save(model.state_dict(), folder + '/epoch_' + str(epoch) +'.pt')
			
		#np.savetxt(folder + '/MAE_test.txt',np.asarray(MAE_test_all))
		np.savetxt(folder + '/MAE_train.txt',np.asarray(MAE_train_all))


