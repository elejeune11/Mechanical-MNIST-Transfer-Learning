import numpy as np
import os
##########################################################################################
#	SUMMARY
##########################################################################################
#####################################
#	FEA problem statement  
#####################################
# We run a finite element simulation where the bottom of the domain is fixed (Dirichlet 
#	boundary condition), the left and right edges of the domain are free, and the top of 
#	the domain is moved to a set of given fixed displacements. In keeping with the size of 
#	the MNIST bitmap ($28 \times 28$ pixels), the domain is a $28 \times 28$ unit square. 
#	We prescribe displacement at the top of the domain up to $50 \%$ of the initial domain 
#	size. The applied displacements $d$ are:
# 			d = [0.0, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0 ] 
#	and data is generated at each displacement step. We run all finite element simulations
#	using the FEniCS computing platform. Mesh refinement studies were conducted, and we 
#	determined that a mesh with $39,200$ quadratic triangular elements is sufficient to 
#	capture the converged solution while not needlessly using computational resources. 
#	This mesh corresponds to $50$ quadratic triangular elements per bitmap pixel. 
#
# To convert the MNIST bitmap images to material properties, we divide the material domain 
#	such that it corresponds with the grayscale bitmap and then specify $E$ as 
#			E = \frac{b}{ 255.0} \, (100.0 - 1.0) + 1.0
#	where $b$ is the corresponding value of the grayscale bitmap that can range from 
#	$0-255$. Poisson's ratio is kept fixed at $\nu = 0.3$ throughout the domain. This 
#	strategy means that the Mechanical MNIST material domains contain a soft background 
#	material with ``digits'' that are two orders of magnitude stiffer. 

##########################################################################################
#	Full dataset can be found @ https://open.bu.edu/handle/2144/39371
##########################################################################################
# for the transfer learning paper, we use only the multi-fidelity data subset
# The cases considered are as follows:
# 	*UE: uniaxial extension, full fidelity dataset (fully refined mesh, quadratic triangular elements, applied displacement is 50% of a side length)
# 	*EE: equibiaxial extension, full fidelity dataset 
# 	*UE-CM-28: uniaxial extension, $28 \times 28 \times 2$ linear triangular elements  
# 	*UE-CM-14: uniaxial extension, $14 \times 14 \times 2$ linear triangular elements
# 	*UE-CM-7: uniaxial extension, $7 \times 7 \times 2$ linear triangular elements
# 	*UE-CM-4: uniaxial extension,  $4 \times 4 \times 2$ linear triangular elements
# 	*UE-perturb: uniaxial extension, applied displacement is a perturbation (.001 units)
# 	*UE-CM-28-perturb: uniaxial extension, $28 \times 28 \times 2$ linear triangular elements, applied displacement is a perturbation (.001 units)
# For all cases, there are 60,000 training examples and 10,000 test examples  
##########################################################################################
# info to set 
##########################################################################################
# specify the data filenames: (could just loop through all of these)
fname = 'UE'
# fname = 'EE'
# fname = 'UE_CM_28'
# fname = 'UE_CM_14'
# fname = 'UE_CM_7'
# fname = 'UE_CM_4'
# fname = 'UE_perturb'
# fname = 'UE_CM_28_perturb'

is_perturb = False

# specify the file location
folder = '../sample_data/'

# specify how many samples ot use for training 
mini_size_train_list = [100,1000] 
# used for the paper: 
# mini_size_train_list = [100,1000,10000,60000]
 
for zzzz in range(0,len(mini_size_train_list)):

	mini_size_train = mini_size_train_list[zzzz]
	mini_size_test = 10000 # specify size of test dataset

	# --> make folder
	if not os.path.exists('NPY_ARRAYS'):
		os.mkdir('NPY_ARRAYS')

	##########################################################################################
	# import data
	##########################################################################################
	MNIST_bitmap_train = np.loadtxt(folder + 'mnist_img_train.txt').astype(np.uint8)
	MNIST_bitmap_test = np.loadtxt(folder + 'mnist_img_test.txt').astype(np.uint8)
	##########################################################################################
	if is_perturb:
		final_psi_train = np.loadtxt(folder + fname + '_psi_train.txt')[:]*10**8
		final_psi_test  = np.loadtxt(folder + fname + '_psi_test.txt')[:]*10**8
	else:
		final_psi_train = np.loadtxt(folder + fname + '_psi_train.txt')[:]
		final_psi_test  = np.loadtxt(folder + fname + '_psi_test.txt')[:]
	##########################################################################################
	# --> save everything 
	# --> MNIST bitmaps
	MNIST_bitmap_train = MNIST_bitmap_train.astype(np.uint8)
	MNIST_bitmap_test = MNIST_bitmap_test.astype(np.uint8)
	np.save('NPY_ARRAYS/' + fname + '_num_' + str(mini_size_train) + '_MNIST_bitmap_train.npy',MNIST_bitmap_train[0:mini_size_train,:])
	np.save('NPY_ARRAYS/' + fname + '_num_' + str(mini_size_test) + '_MNIST_bitmap_test.npy',MNIST_bitmap_test[0:mini_size_test,:])

	# --> change in free energy i.e. QoI 
	np.save('NPY_ARRAYS/' + fname + '_num_' + str(mini_size_train) + '_final_psi_train.npy',final_psi_train[0:mini_size_train])
	np.save('NPY_ARRAYS/' + fname + '_num_' + str(mini_size_test) + '_final_psi_test.npy',final_psi_test[0:mini_size_test])
