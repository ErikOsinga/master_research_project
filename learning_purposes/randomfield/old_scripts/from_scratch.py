from __future__ import print_function, division
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf


"""
From:
http://andrewwalker.github.io/statefultransitions/post/gaussian-fields/
"""

def fft_Indgen(n): # index generator
	"""
	Generates indices for size n
	returns a list which is range(0,n//2+1) and range(-n//2,0) appended
	"""
	a = range(0, n//2+1)
	b = range(1, n//2)
	b.reverse()
	b = [-i for i in b]

	return a + b

def gaussian_random_field(Pk = lambda k : k**-3.0, size = 100):
	"""
	Pk is power spectrum
	size is the size of the image
	"""

	def Pk2(kx, ky): #Pk2 is..
		if kx == 0 and ky == 0:
			return 0.0
		return np.sqrt(Pk(np.sqrt(kx**2 + ky**2)))

	# compute 2D Fourier Transform of gaussian random noise
	noise = np.fft.fft2( np.random.normal(size= (size,size)))

	# Store amplitude of Power spectrum in this array
	amplitude = np.zeros((size,size))
	all_kx = []
	all_ky = []
	for i, kx in enumerate(fft_Indgen(size)):
		for j, ky in enumerate(fft_Indgen(size)):
			amplitude[i,j] = Pk2(kx,ky)
	
	# compute inverse 2D Fourier Transform
	return np.fft.ifft2( noise * amplitude), noise, amplitude


def compute_power_spectrum(data):
	np.fft.fft2(data)


def main1():
	for alpha in [1,2,-4.0, -3.0, -20, -2.0]:
		size = 100
		Pk = lambda k: k**alpha

		out, noise, amplitude = gaussian_random_field(Pk = Pk, size=100)
		fig, ax = plt.subplots(3,1,figsize=[12,12],gridspec_kw = {'width_ratios':[1], 'height_ratios':[1,1,1]})
		plt.subplots_adjust(hspace = 0.5)

		# ax[0].plot(Pk(np.linspace(1,5,10)))
		# ax[0].set_yscale('log')
		fig.suptitle(r'Power spectrum = $k^\alpha$ with $\alpha$ = '+str(alpha))

		ax[0].imshow(noise.real,interpolation='none')
		ax[0].set_title('Draws from 1D Gaussian in Fourier space')
		ax[0].set_xticks([])
		ax[0].set_yticks([])

		real_x = fft_Indgen(size)
		real_y = fft_Indgen(size)
		# dx = (real_x[1] - real_x[0]) / 2
		# dy = (real_y[1] - real_y[0]) / 2
		# extent = [real_x[0]-dx, real_x[-1]+dx, real_y[0]-dy, real_y[-1]+dy]

		im = ax[1].imshow(amplitude,interpolation='none',vmin=0,vmax=np.percentile(amplitude,95))
		# print (len(ax[1].get_xticks()))
		# ax[1].set_xticks(ax[1].get_xticks())
		# ax[1].set_yticks(ax[1].get_yticks())
		# ax[1].set_xticklabels(real_x[::15])
		# ax[1].set_yticklabels(real_y[::15])
		ax[1].set_title('Array: Amplitude of power spectrum')
		ax[1].set_xticks([])
		ax[1].set_yticks([])
		fig.colorbar(im,ax=ax[1])

		ax[2].imshow(out.real,interpolation='none')
		ax[2].set_title('Gaussian random field from IFFT (power spectrum * noise)')
		ax[2].set_xticks([])
		ax[2].set_yticks([])
		plt.savefig('Figures/randomfield_alpha_'+str(alpha)+'.png')
		# plt.show()
		plt.close('all')

		# compute_power_spectrum(out)

main1()


# from http://kmdouglass.github.io/posts/correlated-noise-and-the-fft.html
def main2():
	"""This example considers Perlin Noise, which is correlated random noise
	The algorithm provides two independent (real and imag part) and random signals that are correlated
	over distances comparable to sigma_f. At distances larger than sigma_f the signal is uncorrelated
	The strength of the fluctuations is determined by sigma_r, which is the sqrt
	of the variance of the underlying uncorrelated random signal

	"""

	M = 2**10 # size of the 1D grid. (1024)
	L = 10 # physical size of the grid
	dx = L/M # sampling period
	fs = 1/dx # sampling frequency
	df = 1/L # spacing between frequency components

	# x data from -5 to 5, sampled with M points 
	x = np.linspace(-L/2, L/2, num=M, endpoint=False)
	'''
	# or equivalent
	x = np.fft.fftshift( np.fft.fftfreq(n=M,d=1/dx) )
	'''

	# frequency data from -1/(10*2) to 1/(10*2), samped with M points
	f = np.linspace(-fs/2, fs/2, num=M, endpoint=False)
	'''
	# or equivalent
	x = np.fft.fftshift( np.fft.fftfreq(n=M,d=1/df))
	'''

	sigma_f = 0.1
	sigma_r = 1

	# Define a Gaussian function (provides correlation) in frequency space
	F = np.fft.fftshift(np.exp(-np.pi**2 * sigma_f**2 * f**2)) # FT of Gaussian
	# and an uncorrelated random function in frequency space
	R = np.random.randn(f.size) + 1j * np.random.randn(f.size)

	plt.plot(R,label='uncorrelated random function')
	plt.plot(F,label='Gaussian function')
	plt.title('Frequency space')
	plt.legend()
	plt.show()

	# Convert the product of the two functions back to real space
	noise = 2 * np.pi * np.fft.ifftshift(np.fft.ifft(F * R)) * sigma_r / dx / np.sqrt(df)
	# essentially convolution in the real space of random noise with a gaussian blur

	# Plot the real and imaginary parts of the noise signal
	plt.plot(x, np.real(noise), linewidth = 2, label = 'real')
	plt.plot(x, np.imag(noise), linewidth = 2, label = 'imaginary')
	plt.xlabel('Spatial position')
	plt.grid(True)
	plt.legend(loc = 'best')
	plt.show()




