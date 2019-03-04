import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.stats 
from astropy.cosmology import Planck15 

"""
For generating the redshift distribution of galaxies in tomographic source_photo_z_bins n(z)
for a Euclid-like survey. Following https://arxiv.org/abs/1607.01761

n(z) = dN/dz = the amount of galaxies expected to find in a redshift bin dz

"""

def histedges_equalN(x, nbin):
    npt = len(x)
    return np.interp(np.linspace(0, npt, nbin + 1),
                     np.arange(npt),
                     np.sort(x))

def sigma_z(z): # scatter in photometric redshift
	return sigma*(1+z)

def source_redshift_distribution(z):
	"""
	Assumed dN/dz for Euclid sources. 
	"""
	alpha = 1.3
	beta = 1.5
	z0 = 0.65
	sigma_e = 0.26 # shape noise in each ellipticity component
	dN_dz = A * z**alpha * np.exp(-z/z0)**beta

	return dN_dz

def lens_redshift_distribution(z):
	"""
	Assumed dn/dz for Euclid lenses
	"""
	dn_dz = B * Planck15.comoving_distance(z)**2 * 1/Planck15.H(z)
	return dn_dz.value  # units completely garbage

def pdf_photo_z(z_ph,z):
	"""
	Probability of measuring photo-z z_ph given a galaxy with true redshift z
	"""
	return scipy.stats.norm.pdf(z_ph,loc=z,scale=sigma_z(z))
	
	# same thing but written out
	return 1/np.sqrt(2*np.pi*sigma_z(z)**2)*np.exp(-(z_ph - z - delta_z)**2 
													/ (2*sigma_z(z)**2) )
def cdf_photo_z(z_ph_1,z_ph_2,z):
	"""
	Integrated Probability of measuring photo-z z_ph given a galaxy with true redshift z
	Integrated from z_ph_1 to z_ph_2
	"""
	return 0.5 * (scipy.stats.norm.cdf(z_ph_2,loc=z,scale=sigma_z(z)) 
					 - scipy.stats.norm.cdf(z_ph_1,loc=z,scale=sigma_z(z)) )


def fix_constants():
	"""
	Calculate the constants for the n(z) functions such that expected amount
	of sources per area is correct.
	"""
	global A,B

	print ('Fixing the constants....')

	# calculate n_source = integral from 0 to inf, to fix the constant
	n_source_1 = integrate.quad(source_redshift_distribution,0,np.inf)[0]
	print ( '\nInitially, n_source is %.2f per arcmin^2 integrated from 0 to inf'%(n_source_1) )
	print ('Setting a constant such that n_source = %.2f per arcmin^2'%n_lens)
	A = n_source/n_source_1 # set A such that n_source = 30 per arcmin^2
	n_source_1 = integrate.quad(source_redshift_distribution,0,np.inf)[0]
	print ('Now, n_source is %.2f per arcmin^2 from 0 to inf'%(n_source_1))

	# calculate n_lens(z), to fix the constant
	n_lens_1 = integrate.quad(lens_redshift_distribution,0,np.inf)[0]
	print ( '\nInitially, n_lens is %.2f per arcmin^2 integrated from 0 to inf'%(n_lens_1) )
	print ('Setting a constant such that n_lens = %.2f per arcmin^2'%n_lens)
	B = n_lens/n_lens_1 # set B such that n_lens = 0.25 per arcmin^2
	n_lens_1 = integrate.quad(lens_redshift_distribution,0,np.inf)[0]
	print ('Now, n_lens is %.2f per arcmin^2 integrated from 0 to inf \n'%(n_lens_1))

	return A, B

def create_bins(equal_width):
	"""
	Calculate the photo-z bins used for the source distribution

	equal_width = True, split sample into 'bins' amount of equal with binds
	equal_width = False, split sample into equal area bins.
					(i.e., number of sources in every bin is roughly the same)

	"""

	if equal_width: 
		print ('Using equal width source source_photo_z_bins from 0-%.2f'%z_end)
		_, source_photo_z_bins, _ = plt.hist(z,bins=source_bins)
		plt.close('all')
		# [0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0]

	else: # equal amount
		print ('Using equal amount of sources in every bin from 0-%.2f'%z_end)

		# equal amount of sources means equal area under PDF, so calculate CDF
		cdf = []
		for i in z:
			cdf.append(integrate.quad(source_redshift_distribution,0,i)[0])
		cdf = np.asarray(cdf)

		# total area under z = 0 to z = 2
		total_area = cdf[-1]
		# split into 10(bins) equal area
		area_per_bin = total_area/source_bins

		source_photo_z_bins = []
		source_photo_z_bins.append(0) # start at 0
		for i in range(1,source_bins):
			# find position where CDF = first area
			index = np.where(cdf > i*area_per_bin)[0][0]
			source_photo_z_bins.append(z[index]) # redshift where that happens is bin edge
		
		# last bin ends at last z (= 2)
		index = len(cdf)-1 
		source_photo_z_bins.append(z[index])

		return source_photo_z_bins

def calculate_integral(bins):
	"""
	Calculate the integral of Eq. 7, given the photo-z bins as the list 'bins'
	bins is defined equivalently to how plt.hist() defines the bins
	"""

	# calculate integral over the photo-z bins
	integral_pdf_photoz = []
	for i in range(len(bins)):
		if i == (len(bins) - 1):
			break
		else:
			bin_lower = bins[i]
			bin_upper = bins[i+1]
			integral_pdf_photoz.append(cdf_photo_z(bin_lower,bin_upper,z))
	return integral_pdf_photoz

def calculate_dN_dz_i(integral_pdf_photoz, dN_dz, colors,label,factor):
	"""
	Calculate the final distribution dN/dz per bin, given the 
	value of the integral per bin and the distribution 'dN_dz'
	Plot these as filled curves, with colors given by 'colors'
	
	for the sources:
	dN/dz ~ z**alpha * np.exp(-z/z0)**beta
	for the lenses:
	dN/dz ~ Xi^2/H

	label -- the label that is shown in the plot
	factor -- 10 for the lenses, to make them visible in the plot
				1 for the sources

	"""

	print ('Multiplying %s with factor %.2f'%(label,factor))
	# calculate dN/dz per redshift bin
	dN_dz_i = []
	for i in range(len(integral_pdf_photoz)):
		# compute Eq. 7 
		dN_dz_i.append(dN_dz * integral_pdf_photoz[i])
		# plot as filled curves, fill above y2=0
		plt.fill_between(z,dN_dz_i[i]*factor,y2=0,alpha=0.5,color=colors[i])
		# for assigning a label
		if i == len(integral_pdf_photoz) - 1: 
			plt.fill_between(z,dN_dz_i[i]*factor,y2=0,label=label
				, alpha=0.5,color=colors[i])

	return dN_dz_i
		
### Parameters
delta_z = 0 # bias on photometric redshift
n_source = 30 # per arcmin^2 
n_lens = 0.25 # per arcmin^2 
A = 1. # some constant, to be fixed on the expected n_source
B = 1. # some constant, to be fixed on the expected n_lens
sigma = 0.02 # scatter in photometric redshift = sigma*(1+z)
equal_width = False # determines photo-z source_photo_z_bins, equal width 
					# or equal number of sources in every bin (equal area)
source_bins = 10
lens_bins = 4 # standard
lens_photo_z_bins = [0.2,0.4,0.6,0.8,1.0] # standard

z_end = 2.5
###

# calculate A,B by expected n_source and n_lens
A, B = fix_constants()

# calculate the dN_dz for source and lens
z = np.linspace(0,z_end,1000)
dNsource_dz = source_redshift_distribution(z)
dNlens_dz = lens_redshift_distribution(z)

# Calculate the bins for the sources, if equal_width = False, use equal area
source_photo_z_bins = create_bins(equal_width)

print ('Using the following source redshift bins')
print (source_photo_z_bins)

print('Using the following lens redshift bins:')
print (lens_photo_z_bins)
print('')

# Calculate the integral of Eq. 7 in the paper, using the source_photo_z_bins
source_integral_pdf_photoz = calculate_integral(bins=source_photo_z_bins)
# Calculate the integral of Eq. 7 in the paper, using the lens_photo_z_bins
lens_integral_pdf_photoz = calculate_integral(bins=lens_photo_z_bins)


# Calculate and plot the distribution of sources/lenses per bin
fig = plt.figure(figsize=(8,6))
# generate 10 (bins) shades of blue for the sources
# start at 0.2 because first shades are very light
source_colors = matplotlib.cm.Blues(np.linspace(0.2,1,source_bins))
# generate 4 (bins) shades of red for the lenses
lens_colors = matplotlib.cm.Reds(np.linspace(0,1,lens_bins)) 
# Calculate and plot
dN_source_dz_i = calculate_dN_dz_i(source_integral_pdf_photoz,dNsource_dz,
	source_colors,label='Sources',factor=1)
dN_lens_dz_i = calculate_dN_dz_i(lens_integral_pdf_photoz,dNlens_dz,
	lens_colors,label=r'Lenses $\times$ 10',factor=10)

"""
# calculate dNlens per tomographic (equal_width) bin
dNlens_dz_source_photo_z_bins = []
# generate 10 (bins) shades of red
lens_colors = matplotlib.cm.Reds(np.linspace(0,1,bins)) 
for i in range(len(lens_integral_pdf_photoz)):
	# compute Eq. 7 for the lenses
	dNlens_dz_source_photo_z_bins.append(dNlens_dz * lens_integral_pdf_photoz[i])
	
	# plot as filled curves, fill above y2=0
	plt.fill_between(z,dNlens_dz_source_photo_z_bins[i]*10,y2=0,alpha=0.5, color=lens_colors[i])
	if i == len(lens_integral_pdf_photoz) - 1: # for assigning a label
		plt.fill_between(z,dNlens_dz_source_photo_z_bins[i]*10,y2=0,label=r'Lens tomographic bins $\times$ 10' 
			,alpha=0.5,color=lens_colors[i])
"""	

# Calculate dN_dz or n(z), overall source vs redshift distribution
# as the sum of the individual source_photo_z_bins
dN_dz_final = np.sum(dN_source_dz_i,axis=0)
dN_dz_final += np.sum(dN_lens_dz_i,axis=0)

plt.plot(z,dN_dz_final,label='Total source dist',color='k',alpha=1.0)
plt.xlabel('z',fontsize=16)
plt.ylabel(r'$n(z) = \frac{dN}{d\Omega dz}$',fontsize=16)
plt.legend()
plt.show()




# Load the example file that is given in the cosmosis library
originalfile = '/data1/osinga/cosmosis_installation/cosmosis/cosmosis-standard-library/likelihood/cfhtlens/cfhtlens_heymans13.fits'
hdulist = fits.open(originalfile)
original_nz = Table(hdulist[4].data)

# use the histogram bins from the original file, until z=2.5
hist_bin_lows = original_nz['Z_LOW'][:50]
hist_bin_mids = original_nz['Z_MID'][:50]
hist_bin_highs = original_nz['Z_HIGH'][:50]

for hist_bin in range(72):
	low, mid, high = hist_bin_lows[i], hist_bin_mids[i], hist_bin_highs[i]
	#  make histogram bins of the source bins
	for i in range(source_bins):
		/todo





