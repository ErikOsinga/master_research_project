import numpy as np
import matplotlib.pyplot as plt
import pyccl as ccl

# for running notebooks, plotting inline
# %pylab inline


nz = 1000
z = np.linspace(0.0,2.,nz)

def euclid_ccl(Omega_c, sigma8):
    """
    Generate C_ell as function of ell for a given Omega_c and Sigma8
    Assumes a redshift distribution given by
        z^alpha * exp(z/z0)^beta
        with alpha=1.3, beta = 1.5 and z0 = 0.65

    Assumes photo-z error is Gaussian with a bias is 0.05(1+z)
    """

    cosmo_fid = ccl.Cosmology(Omega_c=Omega_c, Omega_b=0.045, h=0.71, sigma8=sigma8, n_s=0.963)
    ell=np.arange(100,5000)

    pz = ccl.PhotoZGaussian(sigma_z0=0.05)
    dNdz_true = ccl.dNdzSmail(alpha = 1.3, beta = 1.5, z0=0.65)
    
    dNdzs = np.zeros((10, z.size))
    shears = []
    
    for i in range(10):
        # edges of 10 equal width redshift bins, between 0 and 2
        zmin, zmax = i*0.2, (i+1)*.2
        # generate dNdz per bin
        dNdzs[i,:] = ccl.dNdz_tomog(z=z, zmin=zmin, zmax=zmax, pz_func=pz, dNdz_func = dNdz_true)
        # calculate the shear per bin
        gal_shapes = ccl.WeakLensingTracer(cosmo_fid, dndz=(z, dNdzs[i,:]))
        shears.append(gal_shapes)
        
    # calculate 10*9/2 (cross corr) + 10 (autocorr) spectra from the shears
    Cls = []
    for i in range(10):
        for j in range(0,i+1):
            Cls.append(ccl.angular_cl(cosmo_fid, shears[i], shears[j], ell))
     
    return ell, np.array(Cls), dNdzs

ells, Cls, dNdzs = euclid_ccl(0.27, 0.82)

# plot the 55 spectra
fig = plt.figure(figsize=(20,20))
counter = 0
for i in range(10):
        for j in range(0,i+1):
            ax = plt.subplot2grid((10,10), (i,j))
            ax.loglog(ells, ells*(ells+1)*Cls[counter])
            counter += 1

plt.show()


# plot the dNdzs distribution
plt.figure(figsize=(10,10))
for i in range(10):
    plt.plot(z, dNdzs[i,:], lw = 3)

plt.show()


