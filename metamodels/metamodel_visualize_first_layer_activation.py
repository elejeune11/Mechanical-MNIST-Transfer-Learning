"""
Code adapted from: 
@author: Utku Ozbulak - github.com/utkuozbulak Created on Thu Oct 26 11:23:47 2017
"""
import torch
from torch.nn import ReLU
from PIL import Image, ImageFilter
from torchvision import models
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
import os 
import torch.nn as nn
import torch.nn.functional as F
import pickle
##########################################################################################
##########################################################################################
mod_name_list = ['UE_num_100','UE_num_1000','UE_num_100_pretrained_with_UE_CM_28_perturb_num_1000'] 
select_epoch_list = [False, False, True]

for zzzz in range(0,3):
	mod_name = mod_name_list[zzzz]
	select_epoch = select_epoch_list[zzzz]
	
	pretrained_data =  mod_name 

	if select_epoch == True:
		fname = pretrained_data + '/MAE_train.txt' 
		error_all = np.loadtxt(fname)
		arg = np.argmin(error_all)
		pretrained_data = pretrained_data +  '/epoch_' + str(int(arg + 1))

	pretrained_data = pretrained_data + '.pt'
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
	class GuidedBackprop():
		"""
		   Produces gradients generated with guided back propagation from the given image
		"""
		def __init__(self, model):
			self.model = model
			self.gradients = None
			self.forward_relu_outputs = []
			# Put model in evaluation mode
			self.model.eval()
			self.update_relus()
			self.hook_layers()

		def hook_layers(self):
			def hook_function(module, grad_in, grad_out):
				self.gradients = grad_in[0]
			# Register hook to the first layer
			# first_layer = list(self.model.features._modules.items())[0][1]
			first_layer = self.model.conv1
			first_layer.register_backward_hook(hook_function)

		def update_relus(self):
			"""
				Updates relu activation functions so that
					1- stores output in forward pass
					2- imputes zero for gradient values that are less than zero
			"""
			def relu_backward_hook_function(module, grad_in, grad_out):
				"""
				If there is a negative gradient, change it to zero
				"""
				# Get last forward output
				corresponding_forward_output = self.forward_relu_outputs[-1]
				corresponding_forward_output[corresponding_forward_output > 0] = 1
				modified_grad_out = corresponding_forward_output * torch.clamp(grad_in[0], min=0.0)
				del self.forward_relu_outputs[-1]  # Remove last forward output
				return (modified_grad_out,)

			def relu_forward_hook_function(module, ten_in, ten_out):
				"""
				Store results of forward pass
				"""
				self.forward_relu_outputs.append(ten_out)

			# Loop through layers, hook up ReLUs
			# UPDATE ME BASED ON THE ARCHITECTURE OF THE ACTUAL NET 
			pos_li = [0,1]
			mod_li = [self.model.conv1, self.model.conv2]
			for kk in range(0,2):
				pos = pos_li[kk]
				module = mod_li[kk]
				if isinstance(module, ReLU):
					module.register_backward_hook(relu_backward_hook_function)
					module.register_forward_hook(relu_forward_hook_function)

		def generate_gradients(self, input_image, cnn_layer, filter_pos):
			self.model.zero_grad()
			# Forward pass
			x = input_image
			for index, layer in enumerate(list(self.model._modules.items())):
				# Forward pass layer by layer
				# x is not used after this point because it is only needed to trigger
				# the forward hook function
				x = layer[1](x)
				# Only need to forward until the selected layer is reached
				if index == cnn_layer:
					# (forward hook function triggered)
					break
			conv_output = torch.sum(torch.abs(x[0, filter_pos]))
			# Backward pass
			conv_output.backward()
			# Convert Pytorch variable to numpy array
			# [0] to get rid of the first channel (1,3,224,224)
			gradients_as_arr = self.gradients.data.numpy()[0]
			return gradients_as_arr
	##########################################################################################

	##########################################################################################
	# convert to grayscale 
	##########################################################################################
	def convert_to_grayscale(im_as_arr):
		"""
			Converts 3d image to grayscale

		Args:
			im_as_arr (numpy arr): RGB image with shape (D,W,H)

		returns:
			grayscale_im (numpy_arr): Grayscale image with shape (1,W,D)
		"""
		grayscale_im = np.sum(np.abs(im_as_arr), axis=0)
		im_max = np.percentile(grayscale_im, 99)
		im_min = np.min(grayscale_im)
		grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
		grayscale_im = np.expand_dims(grayscale_im, axis=0)
		return grayscale_im
	##########################################################################################
	# save gradient image
	##########################################################################################
	def save_gradient_images(gradient, file_name):
		"""
			Exports the original gradient image

		Args:
			gradient (np arr): Numpy array of the gradient with shape (3, 224, 224)
			file_name (str): File name to be exported
		"""
		if not os.path.exists('results2'):
			os.makedirs('results2')
		# Normalize
		gradient = gradient - gradient.min()
		gradient /= gradient.max()
		# Save image
		path_to_file = os.path.join('results2', file_name + '.jpg')
		save_image(gradient, path_to_file)

	##########################################################################################
	# format np output 
	##########################################################################################
	def format_np_output(np_arr):
		"""
			This is a (kind of) bandaid fix to streamline saving procedure.
			It converts all the outputs to the same format which is 3xWxH
			with using sucecssive if clauses.
		Args:
			im_as_arr (Numpy array): Matrix of shape 1xWxH or WxH or 3xWxH
		"""
		# Phase/Case 1: The np arr only has 2 dimensions
		# Result: Add a dimension at the beginning
		if len(np_arr.shape) == 2:
			np_arr = np.expand_dims(np_arr, axis=0)
		# Phase/Case 2: Np arr has only 1 channel (assuming first dim is channel)
		# Result: Repeat first channel and convert 1xWxH to 3xWxH
		if np_arr.shape[0] == 1:
			np_arr = np.repeat(np_arr, 3, axis=0)
		# Phase/Case 3: Np arr is of shape 3xWxH
		# Result: Convert it to WxHx3 in order to make it saveable by PIL
		if np_arr.shape[0] == 3:
			np_arr = np_arr.transpose(1, 2, 0)
		# Phase/Case 4: NP arr is normalized between 0-1
		# Result: Multiply with 255 and change type to make it saveable by PIL
		if np.max(np_arr) <= 1:
			np_arr = (np_arr*255).astype(np.uint8)
		return np_arr
	##########################################################################################
	# save image
	##########################################################################################
	def save_image(im, path):
		"""
			Saves a numpy matrix or PIL image as an image
		Args:
			im_as_arr (Numpy array): Matrix of shape DxWxH
			path (str): Path to the image
		"""
		if isinstance(im, (np.ndarray, np.generic)):
			im = format_np_output(im)
			im = Image.fromarray(im)
		im.save(path)
	##########################################################################################
	# get_positive_negative_saliency
	##########################################################################################
	def get_positive_negative_saliency(gradient):
		"""
			Generates positive and negative saliency maps based on the gradient
		Args:
			gradient (numpy arr): Gradient of the operation to visualize

		returns:
			pos_saliency ( )
		"""
		pos_saliency = (np.maximum(0, gradient) / gradient.max())
		neg_saliency = (np.maximum(0, -gradient) / -gradient.min())
		return pos_saliency, neg_saliency

	##########################################################################################
	# plot all 
	##########################################################################################
	def plot_all_img(all_img_list, all_title_list,title):
		num_imgs = len(all_title_list)
		num_rows = 10
		num_cols = int(np.ceil(num_imgs/num_rows))
		fig = plt.figure(figsize=(num_cols,num_rows))
		for kk in range(0,num_imgs):
			ax1 = fig.add_subplot(num_rows,num_cols,kk+1)
			ax1.imshow(all_img_list[kk][0,:,:],cmap='bwr')	
			ax1.set_title(all_title_list[kk])
			ax1.axis('off')
			ax1.set_xticklabels([])
			ax1.set_yticklabels([])
			plt.tight_layout()
		#plt.subplots_adjust(wspace=0.1, hspace=0.1)
		plt.savefig(title)
		return
	##########################################################################################
	# actually run 
	##########################################################################################
	for kk in range(1,2): # image indices to use 
		all_img_list = [] 
		all_title_list = [] 

		original_image_fname = '../sample_data/mnist_img_test.txt'
		original_image = np.loadtxt(original_image_fname)[kk,:]
		original_image = original_image*1
		original_image = original_image.reshape(28,28,1)
		################################
		# --> prep image 
		################################
		# mean and std list for channels (Imagenet)
		mean = [0.1306604762738429]
		std = [0.3081078038564622]
	
		im_as_arr = original_image
		im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
		# Normalize the channels
		for channel, _ in enumerate(im_as_arr):
			im_as_arr[channel] /= 255
			im_as_arr[channel] -= mean[channel]
			im_as_arr[channel] /= std[channel]
		# Convert to float tensor
		im_as_ten = torch.from_numpy(im_as_arr).float()
		# Add one more channel to the beginning. Tensor shape = 1,3,224,224
		im_as_ten.unsqueeze_(0)
		# Convert to Pytorch variable
		im_as_var = Variable(im_as_ten, requires_grad=True)

		prep_img = im_as_var
		################################
		################################
		device = torch.device("cpu")
		pretrained_model = Net().to(device)
		pretrained_model.load_state_dict(torch.load(pretrained_data, map_location='cpu'))
		# Guided backprop
		GBP = GuidedBackprop(pretrained_model)

		for cnn_layer in [0,1]:
			if cnn_layer == 0:
				max_amt = 20
			else:
				max_amt = 50
	
			for filter_pos in range(0,max_amt):
				# Get gradients
				file_name_case = 'L' + str(cnn_layer) + '_FP' + str(filter_pos) 
				file_name_to_export = file_name_case 

				guided_grads = GBP.generate_gradients(prep_img, cnn_layer, filter_pos)
				# Save colored gradients
				#save_gradient_images(guided_grads, file_name_to_export + '_Guided_BP_color')
				# Positive and negative saliency maps
				#pos_sal, neg_sal = get_positive_negative_saliency(guided_grads)
				#save_gradient_images(pos_sal, file_name_to_export + '_pos_sal')
				#save_gradient_images(neg_sal, file_name_to_export + '_neg_sal')
				all_img_list.append(guided_grads) 
				all_title_list.append(file_name_case)
				print(file_name_case, 'Complete!')

		title = mod_name + '___%i'%(kk)
		#plot_all_img(all_img_list, all_title_list, title)
		# pickle everything instead -- other script will plot nicely 
		fname1 = title + '_img_list'
		fname2 = title + '_title_list'
		pickle.dump( all_img_list, open( fname1 + ".pkl", "wb" ) )
		pickle.dump( all_title_list, open( fname2 + ".pkl", "wb" ) )

	print('Layer Guided backprop completed')
