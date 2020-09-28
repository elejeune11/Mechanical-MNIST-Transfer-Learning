##########################################################################################
# import necessary modules (list of all simulation running modules)
##########################################################################################
import os
from fenics import *
from dolfin import *
from mshr import *
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
start_time = time.time()
##########################################################################################
# input info / get input file imported 
##########################################################################################
# https://fenicsproject.org/pub/tutorial/html/._ftut1017.html
##########################################################################################
num = int(sys.argv[1]) # command line argument indicates the bitmap to use

is_train = int(sys.argv[2]) == 1 # indicates if it's test or train 

mref = 4 #int(sys.argv[2]) # indicates mesh size -- can set up as command line arg as needed

if is_train:
        fname = 'train_data_' + str(num)
else:
        fname = 'test_data_' + str(num)
        
        
folder_name =  'folder' + '_' + fname + '_' + 'mesh' + str(mref) # output folder  
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# --> import input bitmap
input_folder = '../sample_data' # location of the bitmaps -- may have to update

if is_train:
        line = np.loadtxt(input_folder + '/mnist_img_train.txt')[num,:] # <--MNIST data
else:
        line = np.loadtxt(input_folder + '/mnist_img_test.txt')[num,:]

data_import = line.reshape((28,28))

# --> for laptop data_import = np.loadtxt('input_data/train_data_%i.txt'%(num))

# --> flip input bitmap 
data = np.zeros(data_import.shape)
for jj in range(0,data.shape[0]):
	for kk in range(0,data.shape[1]):
		data[jj,kk] = data_import[int(27.0 - kk),jj] #jj is columns of input, kk is rows

##########################################################################################
# compliler settings / optimization options 
##########################################################################################
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["quadrature_degree"] = 2 #TODO 2
parameters['form_compiler'].add('eliminate_zeros', False)

##########################################################################################
# problem geometry 
##########################################################################################
p_1_x = 0.0; p_1_y = 0.0; p_1_z = 0.0 
p_2_x = 28.0; p_2_y = 28.0; p_2_z = 4.0
nx = int(7*mref) #nx = int(28*mref)
ny =  int(7*mref)#ny = int(28*mref)
nz = int(mref) 
##########################################################################################
# mesh and function spaces 
##########################################################################################
mesh = BoxMesh(Point(p_1_x,p_1_y,p_1_z),Point(p_2_x,p_2_y,p_2_z),nx,ny,nz)

##########################################################################################
# mesh and material prop
##########################################################################################
P2 = VectorElement("Lagrange", mesh.ufl_cell(), 2) #TODO 2
TH = P2
W = FunctionSpace(mesh, TH)
V = FunctionSpace(mesh, 'CG', 1)
back = 1.0
high = 100.0 
nu = 0.3
material_parameters = {'back':back, 'high':high, 'nu':nu}

def bitmap(x,y): #there could be a much better way to do this, but this is working within the confines of ufl
	total = 0   
	for jj in range(0,data.shape[0]):
		for kk in range(0,data.shape[1]):
			const1 = conditional(x>=jj,1,0) # x is rows
			const2 = conditional(x<jj+1,1,0) 
			const3 = conditional(y>=kk,1,0) # y is columns 
			const4 = conditional(y<kk+1,1,0) #less than or equal to? 
			sum = const1 + const2 + const3 + const4
			const = conditional(sum>3,1,0) #ufl equality is not working, would like to make it sum == 4 
			total += const*data[jj,kk]
	return total

class GetMat:
	def __init__(self,material_parameters,mesh):
		mp = material_parameters
		self.mesh = mesh
		self.back = mp['back']
		self.high = mp['high']
		self.nu = mp['nu']
	def getFunctionMaterials(self, V):
		self.x = SpatialCoordinate(self.mesh)
		val = bitmap(self.x[0],self.x[1])
		E = val/255.0*(self.high-self.back) + self.back
		effectiveMdata = {'E':E, 'nu':self.nu}
		return effectiveMdata

mat = GetMat(material_parameters, mesh)
EmatData = mat.getFunctionMaterials(V)
E  = EmatData['E']
nu = EmatData['nu']
lmbda, mu = (E*nu/((1.0 + nu )*(1.0-2.0*nu))) , (E/(2*(1+nu)))
matdomain = MeshFunction('size_t',mesh,mesh.topology().dim())
dx = Measure('dx',domain=mesh, subdomain_data=matdomain)

##########################################################################################
# define boundary domains 
##########################################################################################
btm  =  CompiledSubDomain("near(x[1], btmCoord)", btmCoord = p_1_y)
btmBC = DirichletBC(W, Constant((0.0,0.0,0.0)), btm)

##########################################################################################
# apply traction, and body forces (boundary conditions are within the solver b/c they update)
##########################################################################################
T  = Constant((0.0, 0.0, 0.0))  # Traction force on the boundary
B  = Constant((0.0, 0.0, 0.0))

##########################################################################################
# define finite element problem 
##########################################################################################
u = Function(W)
du = TrialFunction(W)
v = TestFunction(W)

##########################################################################################
# solver loop
##########################################################################################
def problem_solve(th, stretch, u,du,v):	
	expr = Expression(( " cos(th)*(x[0]-L0) - sin(th)*(x[2]-L2) - (x[0]-L0) " , "stretch" , " sin(th)*(x[0]-L0) + cos(th)*(x[2]-L2) - (x[2]-L2)"), th=th, L0=p_2_x/2.0, L2= p_2_z/2.0 , stretch=stretch,degree=2)

	# Updated boundary conditions 
	top  =  CompiledSubDomain("near(x[1], topCoord)", topCoord = p_2_y)
	topBC = DirichletBC(W, expr, top)
	bcs = [btmBC,topBC]

	# Kinematics
	d = len(u)
	I = Identity(d)             # Identity tensor
	F = I + grad(u)             # Deformation gradient
	F = variable(F)
	psi = 1/2*mu*( inner(F,F) - 3 - 2*ln(det(F)) ) + 1/2*lmbda*(1/2*(det(F)**2 - 1) - ln(det(F)))
	f_int = derivative(psi*dx,u,v)
	f_ext = derivative( dot(B, u)*dx('everywhere') + dot(T, u)*ds , u, v)
	Fboth = f_int - f_ext 
	# Tangent 
	dF = derivative(Fboth, u, du)
	#solve(Fboth == 0, u, bcs, J=dF)
	
	# --> alternative solver configuration required for larger meshes 
	problem = NonlinearVariationalProblem(Fboth, u, bcs, dF)
	solver = NonlinearVariationalSolver(problem)
	prm = solver.parameters
 	# Solver parameters
	prm['newton_solver']['linear_solver'] = 'cg'
	# Solve variational problem
	solver.solve()
	
	return u, du, v, f_int, f_ext, psi 

##########################################################################################
# post processing functions 
##########################################################################################
to_print = True
##########################################################################################
def rxn_forces(list_rxn,W,f_int,f_ext):
	x_dofs = W.sub(0).dofmap().dofs()
	y_dofs = W.sub(1).dofmap().dofs()
	z_dofs = W.sub(2).dofmap().dofs() # added for 3D
	f_ext_known = assemble(f_ext)
	f_ext_unknown = assemble(f_int) - f_ext_known
	dof_coords = W.tabulate_dof_coordinates().reshape((-1, 3))  # changed from 2 to 3 for 3D
	y_val_min = np.min(dof_coords[:,1]) + 10E-5; y_val_max = np.max(dof_coords[:,1]) - 10E-5
	x_top = []; x_btm = [] 
	for kk in x_dofs:
		if dof_coords[kk,1] > y_val_max:
			x_top.append(kk)
		if dof_coords[kk,1] < y_val_min:
			x_btm.append(kk)
	f_sum_top_x = np.sum(f_ext_unknown[x_top])
	f_sum_btm_x = np.sum(f_ext_unknown[x_btm])		
	y_top = []; y_btm = [] 
	for kk in y_dofs:
		if dof_coords[kk,1] > y_val_max:
			y_top.append(kk)
		if dof_coords[kk,1] < y_val_min:
			y_btm.append(kk)
	f_sum_top_y = np.sum(f_ext_unknown[y_top])
	f_sum_btm_y = np.sum(f_ext_unknown[y_btm])	
	z_top = []; z_btm = [] # added for 3D
	for kk in z_dofs:
		if dof_coords[kk,1] > y_val_max:
			z_top.append(kk)
		if dof_coords[kk,1] < y_val_min:
			z_btm.append(kk)
	f_sum_top_z = np.sum(f_ext_unknown[z_top])
	f_sum_btm_z = np.sum(f_ext_unknown[z_btm])
	if to_print: 
		print("x_top, x_btm rxn force:", f_sum_top_x, f_sum_btm_x)
		print("y_top, y_btm rxn force:", f_sum_top_y, f_sum_btm_y)
		print("z_top, z_btm rxn force:", f_sum_top_z, f_sum_btm_z)
	list_rxn.append([f_sum_top_x,f_sum_btm_x,f_sum_top_y,f_sum_btm_y,f_sum_top_z,f_sum_btm_z])
	return list_rxn

def pix_centers(u):
	disps_all_x = np.zeros((28,28))
	disps_all_y = np.zeros((28,28))
	disps_all_z = np.zeros((28,28))
	for kk in range(0,28):
		for jj in range(0,28):
			xx = jj + 0.5 # x is columns
			yy = kk + 0.5 # y is rows 
			zz = p_2_z / 2.0
			disps_all_x[kk,jj] = u(xx,yy,zz)[0]
			disps_all_y[kk,jj] = u(xx,yy,zz)[1]
			disps_all_z[kk,jj] = u(xx,yy,zz)[2]
	
	return disps_all_x, disps_all_y, disps_all_z

def strain_energy(list_psi, psi):
	val = assemble(psi*dx)
	list_psi.append(val)
	return list_psi

def strain_energy_subtract_first(list_psi):
	first = list_psi[0]
	for kk in range(0,len(list_psi)):
		list_psi[kk] = list_psi[kk] - first 
	return list_psi
##########################################################################################
##########################################################################################
#fname_paraview = File(folder_name + '/disp.pvd') #<-- primarily for debugging 

list_rxn = []

list_psi = [] 

# --> run the loop
th_val = [0.0, 0.001, 0.01]
str_val = [0.0,0.25/(pi/128)*0.001,0.25/(pi/128)*0.01]

for kk in range(1,17):
	th_val.append(pi/128.0*kk)
	str_val.append(.25*kk)

fname = folder_name + '/pixel_disp' 

for step in range(0,len(th_val)):
	th = th_val[step]
	stretch = str_val[step]
	u, du, v, f_int, f_ext, psi = problem_solve(th,stretch, u,du,v)
	list_rxn = rxn_forces(list_rxn,W,f_int,f_ext)
	#fname_paraview << (u,step)
	disps_all_x, disps_all_y, disps_all_z = pix_centers(u)
	fn_x = fname + '_step' + str(step) + '_x.txt'
	fn_y = fname + '_step' + str(step) + '_y.txt'
	fn_z = fname + '_step' + str(step) + '_z.txt'
	np.savetxt(fn_x,disps_all_x)
	np.savetxt(fn_y,disps_all_y)
	np.savetxt(fn_z,disps_all_z)
	list_psi = strain_energy(list_psi, psi)

# --> save reaction forces
fname = folder_name + '/rxn_force.txt'
np.savetxt(fname,np.asarray(list_rxn))

# --> save total (delta) potential energy 
fname = folder_name + '/strain_energy.txt'
list_psi = strain_energy_subtract_first(list_psi)
np.savetxt(fname, np.asarray(list_psi))	

end_time = time.time()

full_time = end_time - start_time

print('time:')
print(full_time)
	
	
	
	
	
	

