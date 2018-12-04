from __future__ import print_function, division
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf


"""
From:
http://cosmo.nyu.edu/roman/courses/comp_physics_2015/lecture_10.pdf
"""

def fft_Indgen(n): # index generator
	"""
	Generates indices for size n
	returns a list which is range(-n/2,n/2)
	"""
	a = range(0, n//2+1)
	b = range(1, n//2)
	b.reverse()
	b = [-i for i in b]

	return b + a

"""
We start by generating the two-point function in Fourier space: phi(k)
since it is diagonal there. 
It is a complex Gaussian variable independent at each k mode

phi(k) =  A(k) + iB(k)

With A and B real random Gaussian variables with the condition 
A(k) = A(-k) 
B(k) = -B(-k)

"""
def generate_phik(n=100, Pk = lambda k : k**1):
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
	"""
	A_k_array = np.zeros((n,n))
	B_k_array = np.zeros((n,n))
	all_k = np.zeros((n,n))
	for i, kx in enumerate(fft_Indgen(n)):
		for j, ky in enumerate(fft_Indgen(n)):
			# For each k, generate two uniform numbers u1 and u2
			u1 = np.random.uniform()
			u2 = np.random.uniform()
			# Find theta and r
			theta = 2*np.pi*u1
			r = np.sqrt(-2*np.log(1-u2))

			k = np.sqrt(kx**2+ky**2)
			all_k[i,j] = k

			# Build the Fourier coefficients 
			A_k_array[i,j] = np.sqrt(Pk(k)/2) * r * np.cos(theta)
			B_k_array[i,j] = np.sqrt(Pk(k)/2) * r * np.sin(theta)

	# The fourier coefficients below (ky<0) are conjugates of those above (ky>0)
	# For ky = 0, opposite k_x's are also complex conjugates
	# kx = ky = 0 must be a real number, since its is own complex conjugate

	phi_x = np.zeros((n,n))
	for x in range(0,n):
		print (x)
		for y in range(0,n):
			x_vec = np.array([x,y])
			# sum over half of fourier space 
			# seperate ky>0 from ky<0 and for ky=0 seperate kx<0 from kx>0
			for i, kx in enumerate(fft_Indgen(n)):
				for j, ky in enumerate(fft_Indgen(n)):		
					if ky > 0:
						k_vec = np.array([kx,ky])
						phi_x[x,y] += ( A_k_array[i,j] * np.cos(np.dot(k_vec,x_vec))
										 - B_k_array[i,j] * np.sin(np.dot(k_vec,x_vec)) )

					if ky == 0:
						if kx > 0:
							phi_x[x,y] += (A_k_array[i,j] * np.cos(np.dot(k_vec,x_vec))
										 - B_k_array[i,j] * np.sin(np.dot(k_vec,x_vec)) )


	phi_x *= 2
	plt.imshow(phi_x,origin='lower')
	plt.colorbar()
	plt.title('Randomfield from Pk = k^1')
	plt.xlabel('x')
	plt.ylabel('y')
	plt.savefig('./Figures/randomfield_CA.png')
	plt.show()
	plt.close()

	phi_x_test = np.fft.ifft(A_k_array + 1j*B_k_array)
	plt.imshow(phi_x_test.real,origin='lower')
	plt.colorbar()
	plt.title('Randomfield from Pk = k^1 (IFFT.real)')
	plt.xlabel('x')
	plt.ylabel('y')
	plt.savefig('./Figures/randomfield_CA_ifft.png')
	# plt.show()
	plt.close()

	plt.imshow(phi_x_test.imag,origin='lower')
	plt.colorbar()
	plt.title('Randomfield from Pk = k^1 (IFFT.imag)')
	plt.xlabel('x')
	plt.ylabel('y')
	plt.savefig('./Figures/randomfield_CA_ifft_imag.png')
	# plt.show()
	plt.close()

if __name__ == '__main__':
	generate_phik(n=50)