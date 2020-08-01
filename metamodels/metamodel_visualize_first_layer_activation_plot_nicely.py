import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
##########################################################################################
# --> force the center of a colormap onto 0 matplotlib
# Code adapted from: http://chris35wills.github.io/matplotlib_diverging_colorbar/
class MidpointNormalize(colors.Normalize):
	"""
	Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)

	e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
	"""
	def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
		self.midpoint = midpoint
		colors.Normalize.__init__(self, vmin, vmax, clip)

	def __call__(self, value, clip=None):
		# I'm ignoring masked values and all kinds of edge cases to make a
		# simple example...
		x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
		return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))



mod_name_list = ['UE_num_100','UE_num_1000','UE_num_100_pretrained_with_UE_CM_28_perturb_num_1000'] 

for mod_name in mod_name_list:
	num_frames = 1
	
	# ---> find the mins and the max -- normalize all images 
	super_list = []
	for fr in range(0,num_frames):
		title = mod_name + '___%i'%(fr+1)
		fname1 = title + '_img_list'
		fname2 = title + '_title_list'
		all_img_list = pickle.load(open( fname1 + ".pkl", "rb" ) )
		all_title_list = pickle.load(open( fname2 + ".pkl", "rb" ) )
		super_list.append(all_img_list)
	
	max_list = []
	min_list = [] 

	num_imgs = len(all_img_list)
	for jj in range(0,num_imgs):
		ma = []
		mi = [] 
		for fr in range(0,num_frames):
			ma.append(np.max(super_list[fr][jj]))
			mi.append(np.min(super_list[fr][jj]))
			
		max_list.append(np.max(ma))
		min_list.append(np.min(mi))

	for kk in range(0,num_imgs):
		max_list.append(np.max(all_img_list[kk]))
		min_list.append(np.min(all_img_list[kk]))

	for fr in range(0,num_frames):
		all_img_list = super_list[fr]

		num_imgs = 20 #len(all_title_list)
		num_rows = 1
		num_cols = int(np.ceil(num_imgs/num_rows))
		fig = plt.figure(figsize=(num_cols,num_rows))
		for kk in range(0,num_imgs):
			ax1 = fig.add_subplot(num_rows,num_cols,kk+1)
			ima = all_img_list[kk][0,:,:]
			a_min = min_list[kk]
			a_max = max_list[kk]
			ax1.imshow(ima, cmap='afmhot',clim=(a_min, a_max), norm=MidpointNormalize(midpoint=0.0,vmin=a_min, vmax=a_max))
			ax1.axis('off')
			ax1.set_xticklabels([])
			#plt.tight_layout()
		title = mod_name + '__strip_%i'%(fr+1)
		plt.savefig(title)
		plt.close()










