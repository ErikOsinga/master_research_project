import sys
import numpy as np
import matplotlib.pyplot as plt

import pyccl as ccl 


def euclid_ccl(Omega_c, sigma8=0.83):
    """
    Generate C_ell as function of ell for a given Omega_c and Sigma8

    Inputs
        Omega_c -- float: CDM density 
        Sigma_8 -- float: sigma_8

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

    cosmo_fid = ccl.Cosmology(Omega_c=Omega_c, Omega_b=0.045, h=0.71, sigma8=sigma8, n_s=0.963)

    dNdzs = np.zeros((nbins, z.size))
    shears = []
    
    for i in range(nbins):
        # edges of 1 equal width redshift bins, between 0 and 2
        zmin, zmax = i*(2./nbins), (i+1)*(2./nbins)
        # generate dNdz per bin
        dNdzs[i,:] = ccl.dNdz_tomog(z=z, zmin=zmin, zmax=zmax, pz_func=pz, dNdz_func = dNdz_true)
        # calculate the shear per bin
        gal_shapes = ccl.WeakLensingTracer(cosmo_fid, dndz=(z, dNdzs[i,:]))
        shears.append(gal_shapes)
       
    print ('Shears calculated, calculating power spectra now...')
    # calculate nbin*(nbin+1)/2 = 1 spectra from the shears
    Cls = []
    for i in range(nbins):
        for j in range(0,i+1):
            Cls.append(ccl.angular_cl(cosmo_fid, shears[i], shears[j], ells))
     
    return np.array(Cls), dNdzs


nz = 1000 #redshift resolution
zmin = 0.
zmax = 2.
z = np.linspace(zmin,zmax,nz)
# number of tomographic bins
nbins = 1 
# number of cross/auto angular power spectra
ncombinations = int(nbins*(nbins+1)/2)
# 100 log equal spaced ell samples 
ells = np.logspace(np.log10(100),np.log10(1000),100)
"""
Assume a redshift distribution given by
    z^alpha * exp(z/z0)^beta
    with alpha=1.3, beta = 1.5 and z0 = 0.65
"""
dNdz_true = ccl.dNdzSmail(alpha = 1.3, beta = 1.5, z0=0.65)
# Assumes photo-z error is Gaussian with a bias is 0.05(1+z)
pz = ccl.PhotoZGaussian(sigma_z0=0.05)

Omega_c = 0.21
print (f'Generating power spectra with Omega_c = {Omega_c}')
Cls, dNdzs = euclid_ccl(Omega_c)

Omega_c = 0.20
print (f'Generating power spectra with Omega_c = {Omega_c}')
Cls, dNdzs = euclid_ccl(Omega_c)