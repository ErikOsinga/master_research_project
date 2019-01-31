import numpy as np
import matplotlib.pyplot as plt
import pyccl as ccl

# for running notebooks, plotting inline
# %pylab inline


nz = 1000
zmin = 0.
zmax = 2.
z = np.linspace(zmin,zmax,nz)
nbins = 3

def euclid_ccl(Omega_c, sigma8):
    """
    Generate C_ell as function of ell for a given Omega_c and Sigma8
    Assumes a redshift distribution given by
        z^alpha * exp(z/z0)^beta
        with alpha=1.3, beta = 1.5 and z0 = 0.65

    Assumes photo-z error is Gaussian with a bias is 0.05(1+z)
    """

    cosmo_fid = ccl.Cosmology(Omega_c=Omega_c, Omega_b=0.045, h=0.71, sigma8=sigma8, n_s=0.963)
    
    ell = np.logspace(np.log10(100),np.log10(6000),10)

    pz = ccl.PhotoZGaussian(sigma_z0=0.05)
    dNdz_true = ccl.dNdzSmail(alpha = 1.3, beta = 1.5, z0=0.65)
    
    dNdzs = np.zeros((nbins, z.size))
    shears = []
    
    for i in range(nbins):
        # edges of nbins equal width redshift bins, between 0 and 2
        zmin, zmax = i*(2./nbins), (i+1)*(2./nbins)
        # generate dNdz per bin
        dNdzs[i,:] = ccl.dNdz_tomog(z=z, zmin=zmin, zmax=zmax, pz_func=pz, dNdz_func = dNdz_true)
        # calculate the shear per bin
        gal_shapes = ccl.WeakLensingTracer(cosmo_fid, dndz=(z, dNdzs[i,:]))
        shears.append(gal_shapes)
        
    # calculate nbin*(nbin+1)/2 spectra from the shears
    Cls = []
    bin_indices = [] # list of length nbin*(nbin+1)/2 containing tuples with the indices of the bins
    for i in range(nbins):
        for j in range(0,i+1):
            bin_indices.append((i,j))
            Cls.append(ccl.angular_cl(cosmo_fid, shears[i], shears[j], ell))
     
    return ell, np.array(Cls), dNdzs, bin_indices

def euclid_nzs(num_dens):
    '''
    euclid num density = 30 arcmin^-2 = 108,000 deg^-2
    In steradians: A steradian is (180/π)^2 square degrees, or 3282.8 deg^2
    So Euclid number density 
    = 108,000 * 3282.8 = 354,543,086 galaxies per steradian
    
    '''
    nz = 1000
    # zmin , zmax = 0., 3.
    # z = np.linspace(zmin, zmax, nz)
    pz = ccl.PhotoZGaussian(sigma_z0=0.05)
    dNdz_true = ccl.dNdzSmail(alpha = 1.3, beta = 1.5, z0=0.65)
    dNdz_obs = ccl.dNdz_tomog(z=z, zmin=zmin, zmax=zmax, pz_func=pz, dNdz_func = dNdz_true)
    # scale to the given number density
    dNdz_obs = dNdz_obs/dNdz_obs.sum() * num_dens
    nzs = []
    for i in range(nbins):
        # calculate the number density of galaxies per steradian per bin
        zmin_i, zmax_i = i*(2./nbins), (i+1)*(2./nbins)
        mask = (z>zmin_i)&(z<zmax_i)
        nzs.append(dNdz_obs[mask].sum())
     
    return nzs

def total_variance_Cls(i, j, m, n, nzs, ell_index, fsky, sn):
    '''
    fsky = fraction of sky, 15000/41252.96 for Euclid
    sn = shape noise = 0.3 ---> sn^2 = 0.3^2
    Cls = angular auto,cross power spectra
    nzs = array of number density values per tomographic bin
    '''
    Modes_per_bin = fsky*(2*ells[ell_index]+1)

    ss_var_ij = (CL[i,m,ell_index]*CL[j,n,ell_index] + CL[i,n,ell_index]*CL[j,m,ell_index])/Modes_per_bin
    
    """
    sn_var_ij = (CL[i,m,ell_index]*Nb_kroneker(j,n, sn) + Nb_kroneker(i,m, sn)*CL[j,n,ell_index] \
                 + CL[i,n,ell_index]*Nb_kroneker(m,j,sn) + Nb_kroneker(i,n,sn)*CL[m,j,ell_index])/Modes_per_bin
    

    nn_var_ij = (Nb_kroneker(i,m,sn)*Nb_kroneker(j,n,sn) + Nb_kroneker(i,n,sn)*Nb_kroneker(j,m,sn))/Modes_per_bin
    """
    nn_var_ij = 0
    sn_var_ij = 0

    return ss_var_ij + sn_var_ij + nn_var_ij

def Modes_per_bin(b, ell_bin_edges, fsky):
    """
    Return the number of modes in the given l bin 'b'
    
    b -- int, label for the l bin
    ell_bin_edges -- array, edges of the l bins
    fsky -- float, fraction of the sky observed
    """

    # when binning, N_mode,b = fsky * (l_max_b^2 - l_min_b^2)
    lb_min = ell_bin_edges[b]
    lb_max = ell_bin_edges[b+1]
    
    return fsky * (lb_max**2 - lb_min**2)

def Nb_kroneker(i,j, sn):
    # shot noise spectrum, zero when i != j
    
    if i == j :
        x = sn/nzs[i]
        return x
    else:
        return 0

def covariance_matrix():

    # in this case, covariance is (60,60)
    covariance = np.zeros((Cls.size,Cls.size))

    counter1 = 0
    counter2 = 0

    index_holder = []
    for i in range(nbins):
        for j in range(0, i+1):
            for ell_index1 in range(0,len(ells)):
            # for every of the 3*10 possible combinations, 

                # mix with 3*10 possible combinations
                counter2 = 0

                temp = []
                for m in range(nbins):
                    for n in range(0, m+1):
                        for ell_index2 in range(0,len(ells)):
                            covariance[counter1,counter2] = total_variance_Cls(
                                            i, j, m, n, nzs, ell_index1, fsky, sn)
                            counter2 += 1

                            index_holder.append((i,j,ell_index1, (m,n,ell_index2)))
                counter1 +=1



if __name__ == '__main__':
    ells, Cls, dNdzs, bin_indices = euclid_ccl(0.27, 0.82)

    # plot the nbins*(nbins+1)/2 spectra
    fig = plt.figure()
    counter = 0
    for i in range(nbins):
            for j in range(0,i+1):
                ax = plt.subplot2grid((nbins,nbins), (i,j))

                # for the legend
                ax.plot(ells[0], ells[0]*(ells[0]+1)*Cls[counter][0]
                    ,color='white',label=f'{i},{j}')

                ax.loglog(ells, ells*(ells+1)*Cls[counter])
                ax.legend(frameon=False,loc='upper left')

                counter += 1

                if i == 0 and j == 0:
                    ax.set_ylabel('$\ell  (\ell + 1) C_\ell$')
                if i == nbins-1 and j == 0:
                    ax.set_xlabel('$\ell$')
                
    # plt.savefig('./C_ell_3bins')
    # plt.show()
    plt.close()

    # plot the dNdzs distribution
    plt.figure()
    for i in range(nbins):
        plt.plot(z, dNdzs[i,:], lw = 3)

    plt.ylabel('dN/dz')
    plt.xlabel('z')
    # plt.savefig('./dNdz_3bins')
    # plt.show()
    plt.close()

    fsky = 15000/41252.96
    sn = 0.26

    # for indexing the C_l's, Cl[i,j,ell_index]
    # i and j are tomographic indices ell_index is index of ell number
    CL = np.zeros((nbins,nbins, Cls.shape[1]))
    counter = 0
    for i in range(nbins):
            for j in range(0,i+1):
                CL[i,j,:] = Cls[counter]
                counter += 1
    # symmetric
    for i in range(nbins):
        for j in range(nbins):
            CL[i,j,:] = CL[j,i,:]

    # TODO: inspect this num_dens and nzs
    num_dens = 30 * 3600 * (180/np.pi)**2# from arcmin^-2 to deg^-2 to sr
    # number density of the galaxies per steradian on the sky in the whole distribution ni(z)
    nzs = euclid_nzs(num_dens) 



