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
ncombinations = int(nbins*(nbins+1)/2)

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
    Calculate the total Gaussian covariance, which is diagonal in ell, follows
    https://arxiv.org/pdf/1809.09148.pdf

    fsky = fraction of sky, 15000/41252.96 for Euclid
    sn = shape noise = 0.3 ---> sn^2 = 0.3^2
    Cls = angular auto,cross power spectra
    nzs = array of number density values per tomographic bin
    '''
    delta_l = 1
    Modes_per_bin = fsky*(2*ells[ell_index]+1)*delta_l

    ss_var_ij = (CL[i,m,ell_index]*CL[j,n,ell_index] + CL[i,n,ell_index]*CL[j,m,ell_index])/Modes_per_bin
    
    
    sn_var_ij = (CL[i,m,ell_index]*Nb_kroneker(j,n, sn) + Nb_kroneker(i,m, sn)*CL[j,n,ell_index] \
                 + CL[i,n,ell_index]*Nb_kroneker(m,j,sn) + Nb_kroneker(i,n,sn)*CL[m,j,ell_index])/Modes_per_bin
    

    nn_var_ij = (Nb_kroneker(i,m,sn)*Nb_kroneker(j,n,sn) + Nb_kroneker(i,n,sn)*Nb_kroneker(j,m,sn))/Modes_per_bin
    
    # nn_var_ij = 0
    # sn_var_ij = 0

    return ss_var_ij + sn_var_ij + nn_var_ij

def Modes_per_bin(b, ell_bin_edges, fsky):
    """
    Return the number of modes in the given l bin 'b'
    
    b -- int, label for the l bin
    ell_bin_edges -- array, edges of the l bins
    fsky -- float, fraction of the sky observed
    """
    raise ValueError("Not used, use (2l+1)*fsky*delta_l")

    # when binning, N_mode,b = fsky * (l_max_b^2 - l_min_b^2)
    lb_min = ell_bin_edges[b]
    lb_max = ell_bin_edges[b+1]
    
    return fsky * (lb_max**2 - lb_min**2)

def Nb_kroneker(i,j, sn):
    """
    Shot noise spectrum 
        zero when i != j
        sn**2/number of galaxies when i == j
    """
    
    if i == j :
        x = sn**2/nzs[i]
    else:
        x = 0
    return x

def covariance_matrix(ell_index):
    """
    Calculate the total Gaussian covariance matrix for a given ell_index
    thus for ell = ells[ell_index]

    We can do this because the Gaussian covariance matrix is diagonal in ell
    Thus, for every ell we have a matrix of size (ncombinations,ncombinations)
    """

    # in this case, covariance is (ncombinations,ncombinations)
    covariance_l = np.zeros((ncombinations,ncombinations))

    index1 = 0 # first index of covariance matrix (row index)
    index2 = 0 # second index of covariance matrix (column index)

    indices = [] # list of length 2*ncombinations containing tuples of tuples 
                 # with the indices of the bins, to check if we did it correctly

    for i in range(nbins):
        for j in range(0, i+1):
                index2 = 0
                for m in range(nbins):
                    for n in range(0, m+1):
                        covariance_l[index1,index2] = total_variance_Cls(
                                        i, j, m, n, nzs, ell_index, fsky, sn)
                        indices.append( ((i,j),(m,n)) )
                            
                        index2 += 1

                index1 +=1

    return covariance_l, indices

def indexed_CL(Cls):
    """
    For indexing the C_l's, Cl[i,j,ell_index]
    i and j are tomographic indices ell_index is index of ell number

    Returns 
        CL -- np.array -- shape (nbins,nbins,len(ells)), to access any bin,ell combination


    """
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

    return CL

def plot_all_spectra():
    """
    Plot the nbins*(nbins+1)/2 spectra

    """
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
                
    # plt.savefig('./C_ell_%i_bins'%nbins)
    # plt.show()
    plt.close()

def plot_dNdzs():
    """
    Plot the dNdzs distribution
    """

    plt.figure()
    for i in range(nbins):
        plt.plot(z, dNdzs[i,:], lw = 3)

    plt.ylabel('dN/dz')
    plt.xlabel('z')
    # plt.savefig('./dNdz_3bins')
    # plt.show()
    plt.close()

def plot_sample_variance_only():
    """
    Plots only the SS term.

    As opposed to the 1 bin case, now we get a covariance matrix for every
    ell instead of just one number for every ell.

    """
    delta_l = 1
    
    # Since it is diagonal in ell, that means for every ell,
    # we have a (ncombinations,ncombinations) matrix
    cov_holder = np.zeros((len(ells),ncombinations,ncombinations))
    
    for ell_index in range(len(ells)):
        cov_l, indices = covariance_matrix(ell_index)
        cov_holder[ell_index] = cov_l

    # ############## Plot the covariance matrices for all ells
    if len(ells) > 10:
        print ('Not gonna plot this many ells')
    else:
        fig, axes = plt.subplots(4, 3)
        i = 0
        j = 0
        for ell_index in range(len(ells)):
            #i runs from 0 to 3 included 
            #j runs from 0 to 2 included
            ax = axes[i,j]
            ax.set_title('logCovariance matrix for ell = %i'%ells[ell_index])
            im = ax.imshow(np.log10(cov_holder[ell_index]))
            

            plt.colorbar(im, ax=ax)

            if j != 2:
                j+=1
            else:
                j=0
                i+=1

    plt.show()

    np.holdup()
    # ############## Plot the errorbars as the diagonal elements
    fig = plt.figure()
    counter = 0
    for i in range(nbins):
            for j in range(0,i+1):
                ax = plt.subplot2grid((nbins,nbins), (i,j))

                # for the legend
                ax.plot(ells[0], ells[0]*(ells[0]+1)*Cls[counter][0]
                    ,color='white',label=f'{i},{j}')
                ax.legend(frameon=False,loc='upper left')

                onesigma = np.sqrt(np.diag(cov_holder))

                ax.errorbar(ells, ells*(ells+1)*Cls[0], yerr=ells*(ells+1)*onesigma,fmt='-',lw=1)


                counter += 1

                if i == 0 and j == 0:
                    ax.set_ylabel('$\ell  (\ell + 1) C_\ell$')
                if i == nbins-1 and j == 0:
                    ax.set_xlabel('$\ell$')


    fig, ax = plt.subplots()
    # for .. 
    onesigma = np.sqrt(np.diag(covariance_SS))
    # for the legend indicating the bin
    ax.plot(ells[0], ells[0]*(ells[0]+1)*Cls[0][0]
        ,color='white',label=f'0,0')
    ax.legend(frameon=False,loc='upper left')

    # plotting the actual power spectrum
    ax.errorbar(ells, ells*(ells+1)*Cls[0], yerr=ells*(ells+1)*onesigma,fmt='-',lw=1)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylabel('$\ell  (\ell + 1) C_\ell$')
    ax.set_xlabel('$\ell$')
    ax.set_title('Sample covariance only')
    plt.show()

    return cov_holder, indices

def plot_all_variances():
    """
    TODO: ADJUST

    """
    raise ValueError("TODO")    
    # All 3 terms
    delta_l = 1

    covariance = covariance_matrix()

    Neffmode = (2*ells+1) * delta_l * fsky

    #  SS term only
    covariance_diag = Cls[0]*Cls[0] + Cls[0] * Cls[0]
    covariance_diag /= Neffmode
    covariance_SS = np.diag(covariance_diag)

    # SN term only 
    Nb = Nb_kroneker(0,0,sn)
    covariance_diag = 4 * Cls[0]*Nb 
    covariance_diag /= Neffmode
    covariance_SN = np.diag(covariance_diag)

    # NN term only
    covariance_diag = 2*Nb*Nb
    covariance_diag /= Neffmode
    covariance_NN = np.diag(covariance_diag)

    del covariance_diag

    # ############## Plot the covariance matrix
    '''
    fig, axes = plt.subplots(1, 2)
    ax = axes[0]
    ax.set_title('logCovariance matrix 3 terms')
    im = ax.imshow(np.log10(covariance))
    plt.colorbar(im, ax=ax)
    
    ax = axes[1]
    ax.set_title('logCovariance matrix SS term')
    im = ax.imshow(np.log10(covariance_SS))
    plt.colorbar(im, ax=ax)
    plt.show()
    '''

    # ############# Plot the diagonals of the covariance matrices
    fig = plt.figure()#figsize=(12,8)
    plt.plot(ells, np.diagonal(covariance_SS),label='SS')
    plt.plot(ells, np.diagonal(covariance_SN),label='SN')
    plt.plot(ells, np.diagonal(covariance_NN),label='NN')
    plt.xlabel('$\ell$')
    plt.ylabel('Value')
    plt.yscale('log')
    plt.title('Diagonal elements of covariance matrix Cov($C^{00},C^{00}$)')
    plt.legend()
    plt.show()


    # ############## Plot the errorbars as the diagonal elements    
    fig, axes = plt.subplots(1, 2)
    
    ax = axes[0]
    onesigma = np.sqrt(np.diag(covariance))
    # for the legend indicating the bin
    ax.plot(ells[0], ells[0]*(ells[0]+1)*Cls[0][0]
        ,color='white',label=f'0,0')
    ax.legend(frameon=False,loc='upper left')

    # plotting the actual power spectrum
    ax.errorbar(ells, ells*(ells+1)*Cls[0], yerr=ells*(ells+1)*onesigma,fmt='-',lw=1)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylabel('$\ell  (\ell + 1) C_\ell$')
    ax.set_xlabel('$\ell$')
    ax.set_title('All 3 terms')

    ax = axes[1]
    onesigma = np.sqrt(np.diag(covariance_SS))
    # for the legend indicating the bin
    ax.plot(ells[0], ells[0]*(ells[0]+1)*Cls[0][0]
        ,color='white',label=f'0,0')
    ax.legend(frameon=False,loc='upper left')

    # plotting the actual power spectrum
    ax.errorbar(ells, ells*(ells+1)*Cls[0], yerr=ells*(ells+1)*onesigma,fmt='-',lw=1)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylabel('$\ell  (\ell + 1) C_\ell$')
    ax.set_xlabel('$\ell$')
    ax.set_title('SS term only')
    
    plt.show()


if __name__ == '__main__':
    ells, Cls, dNdzs, bin_indices = euclid_ccl(0.27, 0.82)

    # for indexing the C_l's, Cl[i,j,ell_index]
    CL = indexed_CL(Cls)

    fsky = 15000/41252.96
    sn = 0.26 

    # TODO: inspect this num_dens and nzs
    num_dens = 30 * 3600 * (180/np.pi)**2# from arcmin^-2 to deg^-2 to sr
    # number density of the galaxies per steradian on the sky in the whole distribution ni(z)
    nzs = euclid_nzs(num_dens) 

    cov_holder, indices = plot_sample_variance_only()
    # plot_all_variances()

