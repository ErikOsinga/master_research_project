'''

I have tried to get randomfield to work on several python versions. 

Python3.6, Python3.4 and Python3.3 
These all failed.

Python 2.7.15 seems to work. However this is not ideal as 
"Python 3.x is the present and future of the language" -- python people.

But more importantly, the IMNN code is run using Python-3.6.6
So this might cause conflicts down the road..

'''

from __future__ import print_function, division

import randomfield
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf


# size of generator along x,y,z axis. 
# corresponds to ra, dec, line-of-sight-direction
nx, ny , nz = 64, 64, 64
spacing = 2.5 # Uniform grid spacing in Mpc/h
seed = 123

def test_generate():
    # no cosmology is given so creates the 'Planck13' model
    # also no power spectrum is given so it loads a default power spectrum
    generator = randomfield.Generator(nx,ny,nz,spacing)

    # Generate a delta-field realization
    # It is calculated at z=0, sampled from a distribution with mean zero
    # and k-space variance proportional to the smoothed power spectrum
    data = generator.generate_delta_field(seed=seed,show_plot=True)
    # If show_plot=True then also shows a y,z slice 

    return generator, data

generator, data = test_generate()

def plot_hist(data):
    ''' 
    We ignore the first two axes of data
    thus require data to have a large size on the last axis
    This is obviously not the way to go, since we assume a very
    large size into the z-axis but just for testing.
    '''
    plt.hist(data[0,0])
    plt.show()

# plot_hist(data)


def test_gaussian_variance():
    """
    Tabulate a Gaussian power spectrum::
        P(k) = P0*exp(-(k*sigma)**2/2)
    corresponding to a real-space Gaussian smoothing of white noise, with
    correlation function::
        xi(r) = P0*exp(-r**2/(2*sigma**2))/(2*pi)**(3/2)/sigma**3
    The corresponding variance integrated over a grid with
    kmin <= kx,ky,kz <= kmax is::
        P0/(2*pi)**(3/2)/sigma**3 *
            (erf(kmax*sigma/sqrt(2))**3 - erf(kmin*sigma/sqrt(2))**3)
    """

    assert nx == ny == nz
    kmin = (2* np.pi) / (spacing * nx) # why?
    kmax = np.pi / spacing # why?
    sigma = 2.5 * spacing # why?
    P0 = 1.23 # ??


    calculated_var = P0 / (2*np.pi)**1.5 / sigma**3 * (
        erf(kmax * sigma / np.sqrt(2))**3 -
        erf(kmin * sigma / np.sqrt(2))**3
        )

    power = np.empty(100, dtype=[('k', float), ('Pk', float)])

    # 100 numbers ranging from kmin to sqrt(3)*kmax
    power['k'] = np.linspace(kmin, np.sqrt(3)*kmax, len(power))
    # the power spectrum as function of these 100 numbers k
    power['Pk'] = P0 * np.exp(-0.5 * (power['k'] * sigma)**2)

    ntrials = 10
    measured_var = 0
    # Give the power spectrum to the generator now
    generator = randomfield.Generator(nx, ny, nz, spacing, power=power)
    for trial in range(ntrials):
        data = generator.generate_delta_field(seed=seed + trial)
        measured_var += np.var(data)
    measured_var /= ntrials

    print ('Measured var: %f'%measured_var)
    print ('Calculated var: %f'%calculated_var)

test_gaussian_variance()
