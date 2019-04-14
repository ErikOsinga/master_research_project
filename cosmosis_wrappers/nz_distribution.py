import sys
import numpy as np
import matplotlib.pyplot as plt
import os
from astropy.io import fits
from astropy.table import Table

"""
Make sure the environment is activated with the command below 
before running the script.

source /net/reusel/data1/osinga/cosmosis_installation/cosmosis/setup-my-cosmosis



"""


def nz_fits():
	"""
	Create fits file with the number density n(z) for cosmosis.
	"""
	
	# F = fits.open(testfile)

	

def test_nz_dist():
	data = Table.read(testfile,hdu='NZ_SAMPLE')
	zmid = data['Z_MID']

	sumtotal = 0
	for binname in ['BIN1','BIN2','BIN3','BIN4','BIN5','BIN6']:
		plt.plot(zmid,data[binname],label=binname)
		sumtotal += np.sum(data[binname]*0.05)

	print (sumtotal) # The total is normalized to 1
	
	plt.legend(frameon=False)
	plt.ylabel('n(z)')
	plt.xlabel('z')
	plt.show()
	plt.close()

if __name__ == "__main__":
	testfile = '/data1/osinga/cosmosis_installation/cosmosis/cosmosis-standard-library/likelihood/cfhtlens/cfhtlens_heymans13.fits'

	test_nz_dist()