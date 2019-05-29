import numpy as np
import matplotlib.pyplot as plt
import pyDOE # Design of Experiments
from pyDOE import lhs

def generate_LHS(params,samples,criterion,scale,loc,dist='unif',iterations=None):
	"""
	Return samples from a Latin Hypercube.

	params    -- int    -- number of parameters
	samples   -- int    -- number of samples to generate
	criterion -- string -- see https://pythonhosted.org/pyDOE/randomized.html
	scale     -- array  -- scale of the distributions for the parameters, respectively.
						   (has to be length params)
	loc       -- array  -- loc of the distributions for the parameters, respectively.
						   (has to be length params)
	dist      -- string -- Which distribution to use
	iterations-- int    -- Number of iterations in the maximin and correlations. Default 5

	Options for criterion:
	"center" or "c": center the points within the sampling intervals
	"maximin" or "m": maximize the minimum distance between points, 
					  but place the point in a randomized location within its interval
	"centermaximin" or "cm": same as "maximin", but centered within the intervals
	"correlation" or "corr": minimize the maximum correlation coefficient

	"""

	lhd = lhs(params, samples, criterion,iterations)

	if dist == 'unif':
		# Transform the design to be uniformly distributed U[scale, scale+loc]
		from scipy.stats.distributions import uniform
		design = uniform(loc, scale).ppf(lhd)

		return design

	else:
		raise ValueError(f"{dist} not implemented")


if __name__ == "__main__":
	# Example.

	params = 2
	samples = 50
	criterion = 'maximin'
	# Uniform for Omega_m between 0.2 and 0.4
	# Uniform for S8 between 0.7 and 0.9
	dist = 'unif'
	scale = np.array([0.2, 0.7])
	loc = np.array([0.2, 0.2])

	# Generate the samples, array of shape (params,samples)
	lhd = generate_LHS(params, samples, criterion, scale, loc)

	print (lhd)

	plt.title("Latin Hypercube (square)")
	plt.scatter(lhd[:,0],lhd[:,1])
	plt.xlabel('$\Omega_m$',fontsize=14)
	plt.ylabel('$S_8$',fontsize=14)
	plt.show()