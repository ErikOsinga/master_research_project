import sys
import numpy as np
import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import tensorflow as tf
# change to the path where the IMNN git clone is located
# new version of IMNN by Tom
sys.path.insert(-1,'../../../../IMNNv2/IMNN/')
import IMNN.IMNN as IMNN # make sure the path to the IMNN is given
import IMNN.ABC.ABC as ABC
import IMNN.ABC.priors as priors

import tqdm
sys.path.insert(-1,'../../../') # change to path where utils_mrp is located
import utils_mrp_v2 as utils_mrp
import set_plot_sizes # set font sizes

# For making corner plots of the posterior
import corner # Reference: https://github.com/dfm/corner.py/


# for some reason pyccl doesnt work on eemmeer
import pyccl as ccl # for generating weak lensing cross power spectra


nbins = 1 # number of tomographic bins
# number of cross/auto angular power spectra
ncombinations = int(nbins*(nbins+1)/2)
nz = 1000 #redshift resolution
zbins = [(0.0, 2.0)]
z = np.linspace(0.0,2.0,nz)
ells = np.logspace(np.log10(10),np.log10(10000),1000)
# I think this is 1
delta_l = 1

dNdz_true = ccl.dNdzSmail(alpha = 1.3, beta = 1.5, z0=0.65)
# Assumes photo-z error is Gaussian with a bias is 0.05(1+z)
pz = ccl.PhotoZGaussian(sigma_z0=0.05)

def euclid_ccl(Omega_c, Omega_b=0.045, sigma8=0.83, n_s=0.963, h=0.71, mps='halofit'):
    """
    Generate C_ell as function of ell for a given Omega_c and Sigma8

    Inputs
        Omega_c -- float: CDM density 
        Sigma_8 -- float: sigma_8
        mps     -- string: model to use for matter power spectrum

    Assumed global variables
        z -- np.array: samples of z
        ells -- np.array: samples of ell
        dNdz_true -- ccl.dNdzSmail: dNdz distribution
        pz -- ccl.PhotoZGaussian: PhotoZ error
        nbins -- the amount of tomographic redshift bins

    Outputs
        Cls -- np.array, shape (nbins*(nbins-1)/2 + nbins, len(ell)): 
                Cross/Auto correlation shear spectra for the tomographic bins
        dNdzs -- np.array, shape (nbins,len(z):
                dNdz per redshift bin, for all redshifts

    """

    cosmo_fid = ccl.Cosmology(Omega_c=Omega_c, Omega_b=Omega_b, h=h, sigma8=sigma8, n_s=n_s, matter_power_spectrum=mps)

    dNdzs = np.zeros((nbins, z.size))
    shears = []
    
    for i in range(nbins):
        # edges of 2 redshift bins
        zmin_b, zmax_b = zbins[i]
        print (f'Redshift bin {i}: {zmin_b} - {zmax_b}')
        # generate dNdz per bin
        dNdzs[i,:] = ccl.dNdz_tomog(z=z, zmin=zmin_b, zmax=zmax_b, pz_func=pz, dNdz_func = dNdz_true)
        # calculate the shear per bin
        gal_shapes = ccl.WeakLensingTracer(cosmo_fid, dndz=(z, dNdzs[i,:]))
        shears.append(gal_shapes)
        
    # calculate nbin*(nbin+1)/2 = 3 spectra from the shears
    Cls = []
    for i in range(nbins):
        for j in range(0,i+1):
            Cls.append(ccl.angular_cl(cosmo_fid, shears[i], shears[j], ells))
     
    return np.array(Cls), dNdzs

def calculate_derivative(param):
	"""
	# Calculate logarithmic derivatives numerically
	# d Ln(C_ell) / d Ln(Omega_c)
	"""
    if param == 'Omega_c':
        dOmega = 0.002
        Omega_c = 0.211
        Cls_t, _ = euclid_ccl(Omega_c = Omega_c+dOmega, Omega_b = 0.045, mps='halofit')            
        logderiv = (np.log(Cls_t) - np.log(Cls)) / dOmega * Omega_c
        
    elif param == 'Omega_b':
        dOmega = 0.002
        Omega_b = 0.045
        Cls_t, _ = euclid_ccl(Omega_c = 0.211, Omega_b = 0.045+dOmega, mps='halofit')            
        logderiv = (np.log(Cls_t) - np.log(Cls)) / dOmega * Omega_b
        
    elif param == 'sigma8':
        dsigma = 0.002
        sigma8 = 0.83
        Cls_t, _ = euclid_ccl(Omega_c = 0.211, Omega_b = 0.045, sigma8=sigma8+dsigma, mps='halofit')            
        logderiv = (np.log(Cls_t) - np.log(Cls)) / dsigma * sigma8
        
    elif param == 'n_s':
        dns = 0.002
        n_s = 0.963
        Cls_t, _ = euclid_ccl(Omega_c = 0.211, Omega_b = 0.045, sigma8=0.83, n_s = n_s+dns, mps='halofit')            
        logderiv = (np.log(Cls_t) - np.log(Cls)) / dns * n_s
        
    elif param == 'h':
        dh = 0.01
        h = 0.71
        Cls_t, _ = euclid_ccl(Omega_c = 0.211, Omega_b = 0.045, sigma8=0.83, h = h+dh, mps='halofit')            
        logderiv = (np.log(Cls_t) - np.log(Cls)) / dh * h       
    
    else:
        print (f'Parameter {param} not implemented yet ')
    
    return logderiv


# Fiducial value Omega_c = 0.211
Omega_c = 0.211
Cls, dNdzs = euclid_ccl(Omega_c = Omega_c, Omega_b = 0.045, mps='halofit')

# d Ln(C_ell) / d Ln(Omega_c)
logderiv_Oc = calculate_derivative('Omega_c') # uses domega = 0.002

# 'normal' derivative wrt Omega_C: d (C_ell)/ d (Omega_c)
domega = 0.002
Cls_t, _ = euclid_ccl(Omega_c = Omega_c+domega, Omega_b = 0.045, sigma8=0.83, mps='halofit')            
deriv_Oc = (Cls_t - Cls) / domega

# derivative d log(Cl) / d(omega)
log1deriv_Oc = (np.log(Cls_t) - np.log(Cls)) / domega


############### LINEAR EXTRAPOLATION ################## 
# dC/dO_c = deriv_Oc[0]
# Omega_c = 0.211
# Want to calculate Omega_c = 0.23, going to do it from extrapolation

print (f'Starting Omega: {Omega_c}')
dOmega = 0.03
Omega_c_interp = Omega_c+dOmega
print (f'Extrapolating to Omega: {Omega_c_interp}')

# Linear extrapolation, y(x) = y0 + (x-x0)*dy/dx
Cls_new_ext = Cls + (Omega_c_interp - Omega_c)*deriv_Oc

# Actual calculation
Cls_new, _ = euclid_ccl(Omega_c = Omega_c_interp, Omega_b = 0.045
                     , sigma8=0.83, mps='halofit')         

def plot_difference_linspace():
	# Plot difference between actual calculation and linear extrapol
	fig, axes = plt.subplots(3,1, figsize=(8,6), sharex=True)
	ax = axes[0]
	ax.plot(ells, Cls_new_ext[0])
	ax.set_ylabel(f'$C_\ell (\Omega_c={Omega_c_interp})$')
	ax.set_title("Linear extrapolation")

	ax = axes[1]
	ax.plot(ells, Cls_new[0])
	ax.set_ylabel(f'$C_\ell (\Omega_c={Omega_c_interp})$')
	ax.set_title("Actual calculation")

	ax = axes[2]
	ax.plot(ells, Cls_new[0] - Cls_new_ext[0])
	ax.set_ylabel('$\Delta C_\ell$')
	ax.set_title("Difference")

	ax.set_xlabel('$\ell$')
	ax.set_xscale('log')
	# ax.set_ylim(-3,3)
	# ax.legend(loc='upper right',frameon=False)
	# plt.savefig('./difference_extrapolation.png')
	plt.show()

def plot_difference_logspace():
	# Plot it in log space

	fig, axes = plt.subplots(3,1, figsize=(8,6),sharex=True)
	ax = axes[0]
	logClsnew_ext = np.log(Cls_new_ext[0])
	ax.plot(ells, logClsnew_ext)
	ax.set_ylabel(f'$C_\ell (\Omega_c={Omega_c_interp})$')
	ax.set_xscale('log')
	ax.set_title('Linear extrapolation')

	ax = axes[1]
	logClsnew = np.log(Cls_new[0])
	ax.plot(ells, logClsnew)
	ax.set_ylabel(f'$C_\ell (\Omega_c={Omega_c_interp})$')
	ax.set_xscale('log')
	ax.set_title('Actual calculation')

	difference = logClsnew - logClsnew_ext
	ax = axes[2]
	ax.plot(ells, difference)
	ax.set_ylabel(' $\Delta C_\ell$')
	ax.set_title("Difference")
	ax.set_xlabel('$\ell$')

	# plt.savefig('logdifference_extrapolation')
	plt.show()
