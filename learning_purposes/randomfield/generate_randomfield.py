from __future__ import print_function, division
import sys

import numpy as np

import matplotlib
# matplotlib.use('agg')
# print ('Using matplotlib "agg" mode')

import matplotlib.pyplot as plt
from scipy.special import erf


"""
From:
http://cosmo.nyu.edu/roman/courses/comp_physics_2015/lecture_10.pdf
"""

"""
We start by generating the two-point function in Fourier space: phi(k)
since it is diagonal there. 
It is a complex Gaussian variable independent at each k mode

phi(k) =  A(k) + iB(k)

With A and B real random Gaussian variables with the condition 
A(k) = A(-k) 
B(k) = -B(-k)

"""
def fftIndgen(n):
		a = range(0, n//2+1)
		b = range(1, n//2)
		b.reverse()
		b = [-i for i in b]
		return a + b

def generate_phik(n=100, alpha=1):
	"""
	Generates the two-point function phi(k) in Fourier space.

	n -- the size of the image

	Generated according to phi(k) = A(k) + iB(k). If we define
		theta = 2pi*u1
		r = sqrt(-2ln(1-u2))
	then 
		A = r cos(theta) 
		B = r sin(theta)
	are Gaussian independent variables with zero mean and unit variance.
	
	Then, since | phi(k) |^2 = A^2(k) + B^2(k)
	we have P(k) ~ | phi(k) |^2 = <A^2(k)> + <B^2(k)

	Then we scale their unit variance to P(k)/2 to satisfy the total variance
	being equal to P(k). In essence:

	A(k) = sqrt(P(k)/2) r cos(theta)
	B(k) = sqrt(P(k)/2) r sin(theta)

	returns 
	phi_x -- the randomfield
	"""
	Pk = lambda k: k**alpha
	A_k_array = np.zeros((n,n))
	B_k_array = np.zeros((n,n))
	# For each k, generate two uniform numbers u1 and u2
	all_u1, all_u2 = np.random.uniform(size=(n,n)), np.random.uniform(size=(n,n))
	all_k = np.zeros((n,n))

	# Find theta and r for all the uniform numbers
	all_theta = 2*np.pi*all_u1
	all_r = np.sqrt(-2*np.log(1-all_u2))

	for i, kx in enumerate(fftIndgen(n)):
		for j, ky in enumerate(fftIndgen(n)):

			k = np.sqrt(kx**2+ky**2)
			all_k[i,j] = k # store all k

			# only use half of Fourier space
			if (ky > 0) or (ky==0 and kx>0): 
				# since ky<0 is complex conjugates of those above
				# and for ky=0 opposite kx must also be complex conjugates
				
				# For each k, generate two uniform numbers u1 and u2
				# Find theta and r (see above the for loop)

				# Build the Fourier coefficients 
				A_k_array[i,j] = np.sqrt(Pk(k)/2) * all_r[i,j] * np.cos(all_theta[i,j])
				B_k_array[i,j] = np.sqrt(Pk(k)/2) * all_r[i,j] * np.sin(all_theta[i,j])

	# Build the complex conjugate fourier coefficients
	for i, kx in enumerate(fftIndgen(n)):
		for j, ky in enumerate(fftIndgen(n)):
			"""
			The fourier coefficients below (ky<0) are conjugates of those above (ky>0)
			For ky = 0, opposite k_x's are also complex conjugates
			kx = ky = 0 must be a real number, since its is own complex conjugate, 
			this is fixed since we initialize the arrays with zeros
			"""

			if ky < 0:
				# for ky < 0 opposite ky is complex conjugate, 
				# i.e., A_(kx,ky) = A_(kx,-ky), B_(kx,ky) = -B_(kx,-ky)
				A_k_array[i,j] = A_k_array[i,-j]
				B_k_array[i,j] = -1 * B_k_array[i,-j]
			
			if ky == 0:
				if kx < 0:
					# ky = 0 opposite k_x are complex conjugate
					A_k_array[i,j] = A_k_array[-i,j]
					B_k_array[i,j] = -1 * B_k_array[-i,j]

	phi_k = A_k_array + 1j*B_k_array
	phi_k = np.absolute(phi_k)
	plt.imshow(phi_k)
	plt.xticks([])
	plt.yticks([])
	plt.title('Phi_k for Pk ~ k^%2.f, NOT SHIFTED TO CENTER = (0,0)'%alpha)
	plt.show()
#	# return phi_k


	# To have the origin (kx,ky) = (0,0) in the center
	# # phi_k = np.fft.fftshift(phi_k)
	# But we don't want to do this

	# Calculate the randomfield
	phi_x = np.zeros((n,n))
	for x_index, x in enumerate(fftIndgen(n)): # x also goes from -49 to +50 in a 100,100 map
		print ('Doing x=',x) # will have to do fftshift though to put x,y = 0,0 in the middle
		for y_index, y in enumerate(fftIndgen(n)):
			x_vec = np.array([x,y])
			# sum over half of fourier space 
			# seperate ky>0 from ky<0 and for ky=0 seperate kx<0 from kx>0
			for i, kx in enumerate(fftIndgen(n)):
				for j, ky in enumerate(fftIndgen(n)):		
					if ky > 0:
						k_vec = np.array([kx,ky])
						phi_x[x_index,y_index] += (A_k_array[i,j] * np.cos(np.dot(k_vec,x_vec))
										 - B_k_array[i,j] * np.sin(np.dot(k_vec,x_vec)) )

					if ky == 0:
						if kx > 0:
							phi_x[x_index,y_index] += (A_k_array[i,j] * np.cos(np.dot(k_vec,x_vec))
										 - B_k_array[i,j] * np.sin(np.dot(k_vec,x_vec)) )


	phi_x = 2 * np.fft.fftshift(phi_x)/phi_x.size # to put (x,y) = (0,0) in center of array
	plt.imshow(phi_x,origin='lower')
	plt.colorbar()
	plt.title('Randomfield from Pk = k^%i'%alpha)
	plt.xlabel('x')
	plt.ylabel('y')
	plt.xticks([])
	plt.yticks([])
	plt.savefig('./Figures/randomfield.png')
	# plt.show()
	plt.close()

	phi_x_ift = np.fft.ifft2(phi_k) # now we dont have to do FFT shift if I understand correctly
	plt.imshow(phi_x_ift.real,origin='lower')
	plt.colorbar()
	plt.title('Randomfield from Pk = k^%i (IFFT.real)'%alpha)
	plt.xlabel('x')
	plt.ylabel('y')
	plt.savefig('./Figures/randomfield_ifft.png')
	# plt.show()
	plt.close()

	plt.imshow(phi_x_ift.imag,origin='lower')
	plt.colorbar()
	plt.title('Randomfield from Pk = k^%i (IFFT.imag)'%alpha)
	plt.xlabel('x')
	plt.ylabel('y')
	plt.savefig('./Figures/randomfield_ifft_imag.png')
	# plt.show()
	plt.close()

	return phi_x

def createAnnularMask(dimx, dimy, center, big_radius, small_radius):
	"""
	From https://stackoverflow.com/questions/49837377/creating-ring-shaped-mask-in-numpy
	Better to use the function radial_profile
	"""

	Y, X = np.ogrid[:dimx, :dimy]
	distance_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

	mask = (small_radius <= distance_from_center) & \
		(distance_from_center <= big_radius)
	
	return mask

def radial_profile(data, center):
	"""
	Calculate radial profile of array 'data', given the center 'center'
	"""
	y, x = np.indices((data.shape))
	r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
	r = r.astype(np.int)

	tbin = np.bincount(r.ravel(), data.ravel())
	nr = np.bincount(r.ravel())
	radialprofile = tbin / nr
	return radialprofile 

def gaussian_random_field(Pk = lambda k : k**-3.0, size = 100):
	"""
	From http://andrewwalker.github.io/statefultransitions/post/gaussian-fields/
	"""
	def Pk2(kx, ky):
		if kx == 0 and ky == 0:
			return 0.0
		return np.sqrt(Pk(np.sqrt(kx**2 + ky**2)))

	noise = np.fft.fft2(np.random.normal(size = (size, size)))
	amplitude = np.zeros((size,size))
	kxky = np.zeros((size,size,2))
	for i, kx in enumerate(fftIndgen(size)):
		for j, ky in enumerate(fftIndgen(size)):			
			kxky[i, j] = (kx,ky)
			amplitude[i, j] = Pk2(kx, ky)

	return np.fft.ifft2(noise * amplitude)

def calculate_power_spectrum_from_grf(alpha= -3.0, size = 100):
	"""
	This function calculates the power spectrum from the function gaussian_random_field
	"""
	Pk = lambda k : k**alpha
	grf = gaussian_random_field(Pk, size)
	phi_x = grf.real
	print ('Sum of randomfield',np.sum(grf))

	# calculate the FT of the image
	phi_k = np.fft.fft2(phi_x)
	# only keep the amplitude, which is the square of the absolute
	phi_k = np.absolute(phi_k)**2/(2*np.pi)**2

	# shift the array such that the zero k-mode is in the center
	# before shift, kx and ky are given by  [range(0,size/2+1) + range(-(size/2-1), 1)]
										# e.g., ranging from 0 to 50 included + -49 to 0 included
	phi_k = np.fft.fftshift(phi_k)
	
	# Then calculate the radial profile of phi_k around the center
	center = (phi_k.shape[0]/2,phi_k.shape[1]/2)
	Power_spectrum = radial_profile(phi_k, center)
	all_k = np.arange(0,len(Power_spectrum),dtype='float')

	fig, axarr = plt.subplots(1, 3, figsize=(16,6))
	axarr[0].imshow(phi_x)
	axarr[0].set_title('Randomfield generated from Pk = k^%i'%alpha)
	axarr[0].set_xticks([])
	axarr[0].set_yticks([])

	axarr[1].imshow(np.log10(phi_k))
	axarr[1].set_title('(log10 of) Amplitude of FT of randomfield')
	axarr[1].set_xticks([])
	axarr[1].set_yticks([])

	theoretical = np.asarray(all_k)**alpha
	# Should fix this later so that we dont have to do this
	# intercept = Power_spectrum[0]

	amplitude = 1 # np.mean(Power_spectrum/theoretical)

	# print ('Amplitude is not correct yet, normalizing by..',amplitude)
	# print ('Std of amplitude wrt theoretical',np.std(Power_spectrum/theoretical))
	# print ('intercept as well, adding ',intercept)

	# Should fix this later so that we have the same amplitude and intercept
	theoretical = amplitude*np.asarray(all_k)**alpha
	# start at 1 because k = 0, Power = 0
	axarr[2].plot(all_k[1:],Power_spectrum[1:])
	axarr[2].plot(all_k[1:], theoretical[1:]
		,label='Pk = %.1f k**%.1f'%(amplitude,alpha), ls='dashed')
	axarr[2].set_xlabel('k')
	axarr[2].set_ylabel('P(k)')
	axarr[2].set_title('Power spectrum inferred')
	axarr[2].legend()
	axarr[2].set_yscale('log')
	plt.show()

if __name__ == '__main__':
	alpha = -3
	n = 200
	calculate_power_spectrum_from_grf(alpha,n)

	"""
	phi_x = generate_phik(n=n, alpha= alpha)

	phi_k = np.fft.fft2(phi_x)
	phi_k = np.absolute(phi_k)

	# fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(14, 6))
	# ax[0,0].hist(phi_k.ravel(), bins=100)
	# ax[0,0].set_title('hist(|phi_k|)')
	# ax[0,1].hist(np.log(phi_k).ravel(), bins=100)
	# ax[0,1].set_title('hist(log(|phi_k|))')
	# im2 = ax[1,0].imshow(phi_x, interpolation="none")
	# ax[1,0].set_title('Randomfield')
	# im1 = ax[1,1].imshow(np.log(phi_k), interpolation="none")
	# ax[1,1].set_title('log(|phi_k|)')
	# plt.colorbar(im1)
	# plt.show()

	# then make concentric rings of radius K and width dK to calculate P(k)
	# k runs from -x/2 to x/2
	Pk = []
	k = []
	dk = 1.0
	center = (phi_k.shape[0]/2,phi_k.shape[1]/2)
	for i in range(0,26):
		small_radius = i
		big_radius = small_radius+dk
		annularmask = createAnnularMask(phi_k.shape[0],phi_k.shape[1]
			,center,big_radius,small_radius)
		# masked_array = np.ma.array(phi_k, mask=annularmask^1)
		Pk.append(np.sum(phi_k[annularmask]))
		k.append(i)

	plt.plot(k,Pk)
	plt.xlabel('k')
	plt.ylabel('Pk')
	plt.title('Power spectrum inferred from randomfield generated from Pk = k^%i'%alpha)
	plt.show()
	"""