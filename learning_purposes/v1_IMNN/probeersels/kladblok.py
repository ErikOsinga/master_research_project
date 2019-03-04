import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
# change to the path where the IMNN git clone is located
sys.path.insert(-1,'../../information_maximiser')
import IMNN # make sure the path to the IMNN is given
import tqdm



def generate_data(theta1, theta2, n_samples, train=False):
	'''
	Fiducial parameter is passed as a list so many simulations can be made
	at once

	Returns: array of shape (n_samples,input_shape,N) draws 
	from multivariate normal dist, where N is the dimension of variables
	(i.e., N = len(mean) )
	'''

	mean = [0,0] 
	cov = [[theta1,0],  # x_0 has variance 1, 
	       [0,theta2]] # x_1 has variance 2, no covariance  

	# e.g: input_shape = [10,10,1], n_samples = 1000, len(mean) = 2
	# multivariate_normal returns shape (10,10,1,1000,2)
	# therefore we move the second to last axis to the first
	# so we get shape (1000,10,10,1,2)

	# but also, since the multivariate Gaussian returns ((shape,2)) always
	# we take all but the last argument of the input shape,
	# which is then required to be 2
	assert input_shape[-1] == 2, "Multivariate Gaussian generates (,2) shaped data"
	if train:
		return np.moveaxis(
					np.random.multivariate_normal(mean, cov,
						size = [1] + input_shape[:-1] + [n_samples]
					)
				, -2 , 0)
	else:
		return np.moveaxis(
					np.random.multivariate_normal(mean, cov,
						size = input_shape[:-1] + [n_samples]
					)
				, -2 , 0)


# 10x10 data points , or just 10 data points?
input_shape = [100,2] #10,1]  I don't think we need ,10,1
# Fiducial parameter and perturbed values just above and below
theta1_fid = 100.
theta2_fid = 10
delta_theta = 8

# set a seed to surpress sample variance
seed = np.random.randint(1e6)
np.random.seed(seed)
t_m1 = generate_data(theta1_fid - delta_theta, theta2_fid, 1, train= True) 

# ##############################################
# In [114]: t_m1.shape                                                           
# Out[114]: (1, 1, 100, 1, 2)

# Meaning of the dimensions: 
# 1 simulation at theta1-, containing 100 datapoints which are 2D 
# but there is 1 buffer axis so (100,1,2) 

# We have the buffer axis because the network wont accept 2D input shapes
# ##############################################

np.random.seed(seed)
t_p1 = generate_data(theta1_fid + delta_theta, theta2_fid, 1, train=True)
# 1 simulation at theta1+, containing 100 datapoints which are 2D 
# but there is 1 buffer axis so (100,1,2) 

# Should subtract these two from eachother to get derivative wrt theta1
difference = t_p1 - t_m1 

xp, yp = t_p1[0,0].T#[:,0]
xm, ym = t_m1[0,0].T#[:,0]

fig, ax = plt.subplots(3, 2, figsize = (15, 10))

						# [:,0] removes the dummy axis
ax[0].plot(*t_p1[0,0].T,'x')
ax[0].set_title('Upper training image')
ax[0].axis('equal')
ax[1].plot(*t_m1[0,0].T,'x')
ax[1].set_title('Lower training image')
ax[1].axis('equal')

# select first simulation and first theta and transpose to get x and y seperately
# x, y = difference[0,0].T 	
x, y = xp-xm,yp-ym 	

# remove dummy axis
# x, y = x[0], y[0]

ax[2].plot(x, y,'x')
ax[2].set_title('Difference between upper and lower training images at theta1');
ax[2].axhline(xmin = 0., xmax = 1., y = 0.
	, linestyle = 'dashed', color = 'black')
ax[2].axis('equal')
plt.show()




# 1 simulation at theta2-, containing 100 datapoints which are 2D 
t_m1 = generate_data(theta1_fid, theta2_fid - delta_theta, 1, train= True) 
# but there is 1 buffer axis so (100,1,2) 
# We have the buffer axis because the network wont accept 2D input shapes

# 1 simulation at theta2+, containing 100 datapoints which are 2D 
t_p1 = generate_data(theta1_fid, theta2_fid + delta_theta, 1, train=True)





'''

# # Testing the multivariate Gaussian
# theta1 = 1
# theta2 = 10
# delta_theta = 0.1

# mean = [0,0]
# cov = [ [theta1,0],
# 		[0,theta2]]

# x, y = np.random.multivariate_normal(mean,cov,500)

# fig, ax = plt.subplots(4, 1, figsize = (15, 10))

# 						# [:,0] removes the dummy axis
# ax[0].plot(x,y,'x')
# ax[0].set_title('Fiducial theta1 and theta2')
# ax[0].axis('equal')

# cov_theta1_p = [[1-delta_theta,0],
# 						[0,10]] 
# cov_theta1_m = ( cov - [[delta_theta,0],
# 						[0,0]] )

# xp, yp = np.random.multivariate_normal(mean,)

# ax[1].plot(*t_m1[0,0].T,'x')
# ax[1].set_title('Lower training image')
# ax[1].axis('equal')

# # select first simulation and first theta and transpose to get x and y seperately
# # x, y = difference[0,0].T 	
# x, y = xp-xm,yp-ym 	

# # remove dummy axis
# # x, y = x[0], y[0]

# ax[2].plot(x, y,'x')
# ax[2].set_title('Difference between upper and lower training images at theta1');
# ax[2].axhline(xmin = 0., xmax = 1., y = 0.
# 	, linestyle = 'dashed', color = 'black')
# ax[2].axis('equal')
# plt.show()


'''