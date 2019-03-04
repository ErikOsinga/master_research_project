import numpy as np
import matplotlib.pyplot as plt
import pyccl as ccl

# Start by setting up a cosmology object  (3 as an example)

# Basic cosmology with mostly default parameters and calculating setting.
cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.67, A_s=2.1e-9, n_s=0.96, 
                      Neff=3.046, Omega_k=0.)

# Cosmology which incorporates baryonic correction terms in the power.
cosmo_baryons = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.67, A_s=2.1e-9, n_s=0.96, 
                              Neff=3.046, Omega_k=0., baryons_power_spectrum='bcm', 
                              bcm_log10Mc=14.079181246047625, bcm_etab=0.5, bcm_ks=55.0)

# Cosmology where the power spectrum will be computed with an emulator.
cosmo_emu = ccl.Cosmology(Omega_c=0.27, Omega_b=0.05, h=0.67, sigma8=0.83, n_s=0.96, 
                          Neff=3.04, Omega_k=0., transfer_function='emulator', 
                          matter_power_spectrum="emu")

# background quantities
z = np.linspace(0.0001, 5., 100)
a = 1. / (1.+z)


# Compute distances
chi_rad = ccl.comoving_radial_distance(cosmo, a) 
chi_ang = ccl.comoving_angular_distance(cosmo,a)
lum_dist = ccl.luminosity_distance(cosmo, a)
dist_mod = ccl.distance_modulus(cosmo, a)


# Plot the comoving radial distance as a function of redshift, as an example.
plt.figure()
plt.plot(z, chi_rad, 'k', linewidth=2)
plt.xlabel('$z$', fontsize=20)
plt.ylabel('Comoving distance, Mpc', fontsize=15)
plt.tick_params(labelsize=13)
# plt.show()
plt.close()

# Compute growth quantities :
D = ccl.growth_factor(cosmo, a)
f = ccl.growth_rate(cosmo, a)

plt.figure()
plt.plot(z, D, 'k', linewidth=2, label='Growth factor')
plt.plot(z, f, 'g', linewidth=2, label='Growth rate')
plt.xlabel('$z$', fontsize=20)
plt.tick_params(labelsize=13)
plt.legend(loc='lower left')
# plt.show()
plt.close()

# The ratio of the Hubble parameter at scale factor a to H0:
H_over_H0 = ccl.h_over_h0(cosmo, a)
plt.figure()
plt.plot(z, H_over_H0, 'k', linewidth=2)
plt.xlabel('$z$', fontsize=20)
plt.ylabel('$H / H_0$', fontsize=15)
plt.tick_params(labelsize=13)
# plt.show()
plt.close()

# For each component of the matter / energy budget, 
# we can get $\Omega_{\rm x}(z)$, the fractional energy density at $z \ne 0$.
OmM_z = ccl.omega_x(cosmo, a, 'matter')
OmL_z = ccl.omega_x(cosmo, a, 'dark_energy')
OmR_z = ccl.omega_x(cosmo, a, 'radiation')
OmK_z = ccl.omega_x(cosmo, a, 'curvature')
OmNuRel_z = ccl.omega_x(cosmo, a, 'neutrinos_rel')
OmNuMass_z = ccl.omega_x(cosmo, a, 'neutrinos_massive')

plt.figure()
plt.plot(z, OmM_z, 'k', linewidth=2, label='$\Omega_{\\rm M}(z)$')
plt.plot(z, OmL_z, 'g', linewidth=2, label='$\Omega_{\Lambda}(z)$')
plt.plot(z, OmR_z, 'b', linewidth=2, label='$\Omega_{\\rm R}(z)$')
plt.plot(z, OmNuRel_z, 'm', linewidth=2, label='$\Omega_{\\nu}^{\\rm rel}(z)$')
plt.xlabel('$z$',fontsize=20)
plt.ylabel('$\Omega_{\\rm x}(z)$', fontsize= 20)
plt.tick_params(labelsize=13)
plt.legend(loc='upper right')
# plt.show()
plt.close()

# ########################### 
#  MATTER POWER SPECTRA AND RELATED QUANTITIES
# ###########################

# define a vector of k values
k = np.logspace(-3, 2, 100)
# use same z values as before
# background quantities
z = np.linspace(0.0001, 5., 100)
a = 1. / (1.+z)


z_Pk = 0.2
a_Pk = 1. / (1.+z_Pk)

Pk_lin = ccl.linear_matter_power(cosmo, k, a_Pk)
Pk_nonlin = ccl.nonlin_matter_power(cosmo, k, a_Pk)
Pk_baryon = ccl.nonlin_matter_power(cosmo_baryons, k, a_Pk)
Pk_emu = ccl.nonlin_matter_power(cosmo_emu, k, a_Pk)

plt.figure()
plt.loglog(k, Pk_lin, 'k', linewidth=2, label='Linear')
plt.loglog(k, Pk_nonlin, 'g', linewidth=2, label='Non-linear (halofit)')
plt.loglog(k, Pk_baryon, 'm', linewidth=2, linestyle=':', label='With baryonic correction')
plt.loglog(k, Pk_emu, 'b', linewidth=2, linestyle = '--', label='CosmicEmu')
plt.xlabel('$k, \\frac{1}{\\rm Mpc}$', fontsize=20)
plt.ylabel('$P(k), {\\rm Mpc^3}$', fontsize=20)
plt.xlim(0.001, 50)
plt.ylim(0.01, 10**6)
plt.tick_params(labelsize=13)
plt.legend(loc='lower left')
plt.show()
plt.close()

R = np.linspace(5, 20, 15)

sigmaR = ccl.sigmaR(cosmo, R)
sigma8 = ccl.sigma8(cosmo)

print("sigma8 =", sigma8)

# ###############
# Can also compute C_ell for galaxy counts, galaxy lensing and CMB lensing
# for autocorrelations or any cross-correlation

z_pz = np.linspace(0.3, 3., 3)  # Define the edges of the photo-z bins.
# array([0.3 , 1.65, 3.  ]) AKA 2 bins
pz = ccl.PhotoZGaussian(sigma_z0=0.05)

def dndz(z,args) :
    return z**2*np.exp(-(z/0.5)**1.5)

redshift_dist=ccl.dNdzFunction(dndz)

# galaxy counts
dNdz_nc = [ccl.dNdz_tomog(z=z, zmin=z_pz[zi], zmax=z_pz[zi+1], pz_func=pz
	, dNdz_func=redshift_dist) for zi in range(0, len(z_pz)-1)]
# galaxy lensing
dNdz_len = [ccl.dNdz_tomog(z=z, zmin=z_pz[zi], zmax=z_pz[zi+1], pz_func=pz
	, dNdz_func=redshift_dist) for zi in range(0, len(z_pz)-1)]

# assume linear bias
bias = 2.*np.ones(len(z))

gal_counts = ([ccl.NumberCountsTracer(cosmo, has_rsd=False,
      dndz=(z, dNdz_nc[zi]), bias=(z, bias))  for zi in range(0, len(z_pz)-1)])

gal_lens = ([ccl.WeakLensingTracer(cosmo, dndz=(z, dNdz_len[zi]))
      for zi in range(0, len(z_pz)-1)])

# tracer objects for CMB lensing
cmb_lens = [ccl.CMBLensingTracer(cosmo, z_source=1089.)]

all_tracers = gal_counts + gal_lens + cmb_lens

ell = np.linspace(1, 2000, 2000)

n_tracer = len(all_tracers)

c_ells = ([[ccl.angular_cl(cosmo, all_tracers[ni], all_tracers[nj], ell) 
            for ni in range(0, n_tracer)] for nj in range(0, n_tracer)])

# 2 galaxy count bins + 2 galaxy lens bins + 1 cmb lens bin = 5 tracers
# np.shape(c_ells) : (5, 5, 2000) 

plt.figure()
plt.loglog(ell, c_ells[0][0], 'k', linewidth=2, label='gg bin 1 auto')
plt.loglog(ell, c_ells[0][3], 'g', linewidth=2, label='g1 x src2')
plt.loglog(ell, c_ells[4][4], 'm', linewidth=2, label='CMB lensing auto')
plt.xlabel('$\ell$', fontsize=20)
plt.ylabel('$C_\ell$', fontsize=20)
plt.xlim(1, 1000)
plt.tick_params(labelsize=13)
plt.legend(loc='lower left')
plt.show()


fig = plt.figure()

for i in range(2,4):
  for j in range(2,4):
    if i >= j: 
      ax = plt.subplot2grid((2,2), ((i-2),(j-2)))

      plt.loglog(ell, c_ells[i][j], label=f'src{i-1} x src{j-1}')
      plt.legend(frameon=False)

      if (i, j) == (2,2):
        plt.ylabel(r'$C_\ell$')
      elif (i,j) == (3,3):
        plt.xlabel(r'$\ell$')

plt.subplots_adjust(wspace = 0.4, hspace = 0.4)

plt.show()



