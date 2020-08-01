from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
use_cuda = False
##########################################################################################
# Model name/info 
##########################################################################################
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
##########################################################################################
##########################################################################################
##########################################################################################
for case_num in [1,2,3]: 
	print('CASE_NUM:',case_num)
	##########################################################################################
	# self comparison 
	##########################################################################################
	# define what to evaluate
	if case_num == 1: # UE model, no pre-training
		in_test_fname = 'NPY_ARRAYS/UE_num_10000_MNIST_bitmap_test.npy'
		out_test_fname = 'NPY_ARRAYS/UE_num_10000_final_psi_test.npy'
		models_fname = ['UE_num_100','UE_num_1000']
		save_name = 'LC_UE'
		save_all_correct = np.zeros((10000,len(models_fname)))
		save_all_predict = np.zeros((10000,len(models_fname)))
	elif case_num == 2: # UE-CM-28-perturb model, no pre-training
		in_test_fname = 'NPY_ARRAYS/UE_CM_28_perturb_num_10000_MNIST_bitmap_test.npy'
		out_test_fname = 'NPY_ARRAYS/UE_CM_28_perturb_num_10000_final_psi_test.npy'
		models_fname = ['UE_CM_28_perturb_num_100','UE_CM_28_perturb_num_1000']
		save_name = 'LC_UE_CM_28_perturb'
		save_all_correct = np.zeros((10000,len(models_fname)))
		save_all_predict = np.zeros((10000,len(models_fname)))
	elif case_num == 3:# UE model, pre-trained on UE-CM-28-perturb w/ 1000 data points
		in_test_fname    = 'NPY_ARRAYS/UE_num_10000_MNIST_bitmap_test.npy'
		out_test_fname   = 'NPY_ARRAYS/UE_num_10000_final_psi_test.npy'
		models_fname = ['UE_num_100_pretrained_with_UE_CM_28_perturb_num_1000']
		save_name = 'TL_LC_UE_pre_train_UE_CM_1000'
		save_all_correct = np.zeros((10000,len(models_fname))) 
		save_all_predict = np.zeros((10000,len(models_fname))) 
		
	for NUM in range(0,len(models_fname)):
		pretrained_model = models_fname[NUM]

		class MechMNISTDataset(Dataset):
			""" mechanical MNIST data set"""
			def __init__(self,transform=None, target_transform=None):
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
				if self.transform is not None:
					img = self.transform(img)
				if self.target_transform is not None:
					lab = self.target_transform(lab)
				sample = (img,lab)
				return sample 

		##########################################################################################
		##########################################################################################
		# MNIST Test dataset and dataloader declaration
		test_loader = DataLoader( MechMNISTDataset(transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])), batch_size=1, shuffle=False)
		# Define what device we are using
		print("CUDA Available: ",torch.cuda.is_available())
		device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
		# Initialize the network
		model = Net().to(device)
		# Load the pretrained model
		if case_num > 2:
			folder = pretrained_model
			fname = folder + '/MAE_train.txt' 
			error_all = np.loadtxt(fname)
			print(error_all.shape)
			arg = np.argmin(error_all)
			print(arg)
			model.load_state_dict(torch.load(pretrained_model + '/epoch_' + str(int(arg + 1)) + '.pt', map_location='cpu'))
		else:
			fname = pretrained_model + '.pt'
			model.load_state_dict(torch.load( fname , map_location='cpu'))
		# Set the model in evaluation mode
		model.eval()
		##########################################################################################
		def return_test_all( model, device, test_loader):
			all_test = [] 
			all_target = []

			for data, target in test_loader:
				# Send the data and label to the device
				data, target = data.to(device), target.to(device)
				output = model(data)
				output = model(data).detach().numpy()[0][0]
				target = target.detach().numpy()[0]
				all_test.append(output)
				all_target.append(target)

			return all_test, all_target
		##########################################################################################
		##########################################################################################
		all_test, all_target = return_test_all( model, device, test_loader)
		save_all_correct[:,NUM] = np.asarray(all_target)[:,0]
		save_all_predict[:,NUM] = np.asarray(all_test)[:]

	##########################################################################################
	##########################################################################################
	np.savetxt(save_name + '_test_correct.txt',np.asarray(save_all_correct))
	np.savetxt(save_name + '_test_predict.txt',np.asarray(save_all_predict))