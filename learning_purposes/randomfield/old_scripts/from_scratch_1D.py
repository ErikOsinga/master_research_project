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

	def Pk2(kx): #Pk2 is..
		if kx == 0:
			return 0.0
		return np.sqrt(Pk(np.sqrt(kx**2)))

	# compute Fourier Transform of gaussian random noise
	noise = np.fft.fft( np.random.normal(size=size) )

	# Store amplitude of Power spectrum in this array
	amplitude = np.zeros((size))
	for i, kx in enumerate(fft_Indgen(size)):
			amplitude[i] = Pk2(kx)
	
	# compute inverse Fourier Transform
	return np.fft.ifft( noise * amplitude), noise, amplitude

def delta_power_spectrum(k):
	if np.abs(k) == 10:
		return 1.0
	return 0.0

def compute_power_spectrum(data):
	np.fft.fft2(data)


def main1():
	for alpha in [1,2,-4.0, -3.0, -20, -2.0]:
		size = 100

		Pk = lambda k: k**alpha

		out, noise, amplitude = gaussian_random_field(Pk = Pk, size=size)
		fig, ax = plt.subplots(3,1,figsize=[12,12],gridspec_kw = {'width_ratios':[1], 'height_ratios':[1,1,1]})
		plt.subplots_adjust(hspace = 0.5)

		# ax[0].plot(Pk(np.linspace(1,5,10)))
		# ax[0].set_yscale('log')
		fig.suptitle('Power spectrum, alpha = '+str(alpha))

		ax[0].plot(noise.real)
		ax[0].set_title('Draws from 1D Gaussian')

		im = ax[1].scatter(fft_Indgen(size),amplitude)
		ax[1].set_title('Array: Amplitude of power spectrum')
		ax[1].set_xlabel('k')
		# fig.colorbar(im,ax=ax[1])

		ax[2].plot(out.real)
		ax[2].set_title('Gaussian random data from power spectrum and noise')
		plt.savefig('Figures/randomdata_alpha_'+str(alpha)+'.png')
		# plt.show()
		plt.close('all')

		# compute_power_spectrum(out)

		# break

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




