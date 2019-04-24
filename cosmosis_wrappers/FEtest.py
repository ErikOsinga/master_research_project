import sys
import numpy as np
import matplotlib.pyplot as plt
import os
from cl_plot import plot_cosmosis_cells

"""
Make sure the environment is activated with the command below 
before running the script.

source /net/reusel/data1/osinga/cosmosis_installation/cosmosis/setup-my-cosmosis


Testing the FrankenEmu

"""

def generate_parameter_ini(save_dir, omega_m, h0, omega_b, tau, n_s
	,alpha, beta, z0, sigz, ngal, bias
	,A_s, omega_k, w, wa):
	"""
	Create .ini file containing the parameters
	"""
	with open(f'{save_dir}/params.ini','w') as file:
		# Parameters and data in CosmoSIS are organized into sections
		# so we can easily see what they mean.
		# There is only one section in this case, called cosmological_parameters
		file.write('[cosmological_parameters]')
		file.write('\n')
		file.write(f'omega_m = {omega_m}')
		file.write('\n')
		file.write(f'h0 = {h0}')
		file.write('\n')
		file.write(f'omega_b = {omega_b}')
		file.write('\n')
		file.write(f'tau = {tau}')
		file.write('\n')
		file.write(f'n_s = {n_s}')
		file.write('\n')
		file.write(f'A_s = {A_s}')
		file.write('\n')
		file.write(f'omega_k = {omega_k}')
		file.write('\n')
		file.write(f'w = {w}')
		file.write('\n')
		file.write(f'wa = {wa}')
		file.write('\n')

		file.write('\n')
		file.write('[number_density_params]\n')
		file.write(f'alpha = {alpha}\n')
		file.write(f'beta = {beta}\n')
		file.write(f'z0 = {z0}\n')
		file.write(f'sigz = {sigz}\n')
		file.write(f'ngal = {ngal}\n')
		file.write(f'bias = {bias}\n')

def remove_additional_data(save_dir):
	"""
	Cosmosis outputs many folders with data
	of which we only use "cosmological_parameters" and "shear_cl"

	Typical file sizes:

	324K	./cmb_cl
	92K	./distances
	8.0K	./cosmological_parameters
	464K	./linear_cdm_transfer
	184K	./shear_cl
	1008K	./matter_power_nl
	960K	./matter_power_lin
	36K	./nz_sample
	3.1M	./

	This function removes the unneeded folders with data and the files used
	to generate the data
	"""

	# Since the rm -rf function is so dangerous, check what we are removing first
	if save_dir[:69] != '/net/reusel/data1/osinga/master_research_project/saved_data/cosmosis/':
		raise ValueError(f"Not going to remove files inside {save_dir}")

	else:
		os.system(f"rm -rf {save_dir}/cmb_cl")
		os.system(f"rm -rf {save_dir}/distances")
		os.system(f"rm -rf {save_dir}/linear_cdm_transfer")
		os.system(f"rm -rf {save_dir}/matter_power_lin")
		os.system(f"rm -rf {save_dir}/matter_power_nl")
		os.system(f"rm -rf {save_dir}/nz_sample")

def write_cosmosis_file(save_dir,nbin,zmax,dz,ell_min,ell_max,n_ell):
	"""
	Write the .ini file that is run with cosmosis to generate the params
	"""
	with open(f'{save_dir}/generate_cells.ini','w') as file:
		# Parameters and data in CosmoSIS are organized into sections
		# so we can easily see what they mean.
		file.write('[runtime]\n')
		file.write('sampler = test\n')
		file.write('\n')
		file.write('[test]\n')
		file.write(f'save_dir={save_dir}\n')
		file.write('fatal_errors=T\n')
		file.write('\n')
		# switch this on to enable an analysis of the pipeline that checks 
		# which parameters are "fast". Set fast_slow = T in [pipeline] to enable
		# this for some fast samplers if you want to.
		file.write('analyze_fast_slow = F\n')
		file.write('\n')
		# The pipeline section contains information that describes the sequence
		# of calculations to be done and what we want out at the end.
		file.write('[pipeline]\n')
		# The list of modules to be run, in this order.  
		# The modules named here must appear as sections below.
		file.write('modules = consistency camb halofit extrapolate_power load_nz  shear_shear\n')
		file.write(f'values = {save_dir}/params.ini\n') # the param file created earlier
		file.write('\n')

		# We can get a little more output during the run by setting some values.
		file.write('quiet=F\n')
		file.write('timing=F\n')
		file.write('debug=F\n')
		file.write('\n')

		# calculate derived cosmological parameters
		file.write('[consistency]\n')
		file.write('file = cosmosis-standard-library/utility/consistency/consistency_interface.py\n')
		file.write('\n')
		# Photo-z bias, not sure how this is used yet
		file.write('[photoz_bias]\n')
		file.write('file = cosmosis-standard-library/number_density/photoz_bias/photoz_bias.py\n')
		file.write('mode=additive\n')
		file.write('sample=nz_sample\n')
		file.write('\n')
		# calculate expansion history, recomb history, CMB power spectra, matter power
		# spectra at low z and and sigma_8
		file.write('[camb]\n')
		file.write('file = cosmosis-standard-library/boltzmann/camb/camb.so\n')
		file.write('mode=all\n')
		file.write('lmax=2500\n')
		file.write('feedback=0\n')
		file.write('\n')
		# computes non-linear power spectrum
		file.write('[halofit]\n')
		file.write('file = cosmosis-standard-library/boltzmann/halofit/halofit_module.so\n')
		file.write('\n')
		# Linear extrapolation of matter power spectrum to higher values of (k)
		# unphysical values but useful for numerical stability
		file.write('[extrapolate_power]\n')
		file.write('file=cosmosis-standard-library/boltzmann/extrapolate/extrapolate_power.py\n')
		file.write('kmax=500.0\n')
		file.write('\n')

		# Load the number density from the Smail distribution
		file.write(f"""[load_nz]
file = cosmosis-standard-library/number_density/smail/photometric_smail.py
nbin = {nbin}
zmax = {zmax}
dz = {dz}
output_section=nz_sample ; This output section name is asked by shear-shear module
""")

		# This module uses the Limber approximation to compute shear-shear 
		# C_ell given the shear kernel (which is derived from the number density
		# and from geometry).
		file.write('[shear_shear]\n')
		file.write('file = cosmosis-standard-library/structure/projection/project_2d.py\n')
		file.write(f'ell_min = {ell_min:.1f}\n')
		file.write(f'ell_max = {ell_max:.1f}\n')
		file.write(f'n_ell = {n_ell}\n')
		file.write('shear-shear = sample-sample\n')
		file.write('verbose = F\n')


def generate_cells(save_dir, nbin, zmax, dz, ell_min, ell_max, n_ell, omega_m
	, alpha=1.3, beta=1.5, z0=0.65, sigz=0.05, ngal=30, bias=0 # redshift params, default Euclid
	, h0=0.72, omega_b=0.04, tau=0.08, n_s=0.96
	, A_s=2.1e-9, omega_k=0.0, w=-1.0, wa=0.0, mpi=False):
	
	"""
	Generate weak lensing Cls using cosmosis with the given parameters
	C_ells are saved into save_dir
	
	"""


	# First make sure the save directory is empty
	# Since the rm -rf function is so dangerous, check what we are removing first
	# if save_dir[:69] != '/net/reusel/data1/osinga/master_research_project/saved_data/cosmosis/':
	# 	raise ValueError(f"Not going to remove files inside {save_dir}")
	# print (f"Removing files inside {save_dir}/")
	# os.system(f"rm -rf {save_dir}/")

	# if not os.path.isdir(save_dir):
	# 	print (f"Creating directory {save_dir} for cosmosis")
	# 	os.mkdir(save_dir)

	# # Generate the ini file with the parameters
	# generate_parameter_ini(save_dir, omega_m, h0=h0, omega_b=omega_b, tau=tau,n_s=n_s
	# ,alpha=alpha, beta=beta, z0=z0, sigz=sigz, ngal=ngal, bias=bias
	# ,A_s=A_s, omega_k=omega_k, w=w,wa=wa)

	# Generate the ini file that will calculate everything with above parameters
	# write_cosmosis_file(save_dir, nbin, zmax, dz, ell_min, ell_max, n_ell)

	# Call cosmosis, calculate c_ells
	if mpi:
		# Test sampler does not support mpi
		os.system(f"mpirun -n 4 cosmosis --mpi {save_dir}/generate_cells.ini")
	else:
		os.system(f"cosmosis {save_dir}/generate_cells.ini")

	# Remove matter power spectra and cmb data etc.
	# remove_additional_data(save_dir)

def load_cells(save_dir,nbin):
	"""
	Load the c_ells generated by cosmosis as a numpy array 
	of shape ((nbin*(nbin+1)/2),n_ell) (i.e., (ncombinations,n_ell))

	Returns
	ells -- sampled ell values
	c_ells -- shear cl values
	"""
	data_dir = f'{save_dir}/shear_cl'
	
	ells = np.loadtxt(f'{data_dir}/ell.txt')
	ncombinations = nbin*(nbin+1)//2
	c_ells = np.empty((ncombinations,len(ells)))
	counter = 0
	for j in range(1,nbin+1): # Cosmosis starts numbering at 1
		for i in range(j,nbin+1): 
			c_ells[counter] = np.loadtxt(f'{data_dir}/bin_{i}_{j}.txt')
			counter += 1

	return ells, c_ells

if __name__ == "__main__":
	# Folder where data is saved
	save_dir = '/net/reusel/data1/osinga/master_research_project/saved_data/cosmosis/generate_cells_test2'
	save_dir = '/net/reusel/data1/osinga/master_research_project/saved_data/cosmosis/frankenemu_test'

	nbin, zmax, dz = 3, 2.0, 0.002
	ell_min, ell_max, n_ell = 50, 3000, 200


	Omega_M = 0.315
	Omega_b_fraction = 0.15653724 # fraction of Omega_M
	
	A_s = 2.1e-9
	Omega_b = Omega_b_fraction * Omega_M
	h = 0.674
	n_s = 0.965
	w0 = -1.03

	alpha=1.3
	beta=1.5
	z0=0.65
	sigz=0.05
	ngal=30
	bias=0 

	# Calculate Cls with cosmosis
	generate_cells(save_dir, nbin, zmax, dz, ell_min, ell_max, n_ell
		, alpha=alpha, beta=beta, z0=z0, sigz=sigz, ngal=ngal, bias=bias
		, omega_m=Omega_M, h0=h,omega_b=Omega_b, n_s=n_s, A_s=A_s,w=w0)

	# Folder that holds the shear_cls
	data_dir = f'{save_dir}/shear_cl'
	plot_cosmosis_cells(data_dir, bins=nbin)

	ells, c_ells = load_cells(save_dir, nbin)
