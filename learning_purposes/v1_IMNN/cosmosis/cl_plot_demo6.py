import sys
import numpy as np
import matplotlib.pyplot as plt


cosmosis_dir = '/data1/osinga/cosmosis_installation/cosmosis/'
data_dir = 'demo6/shear_cl/'


ell = np.loadtxt(f'{cosmosis_dir}{data_dir}ell.txt')

fig = plt.figure()

for i in range(1,7):
	for j in range(1,7):
		if i >= j: 
			c_ell = np.loadtxt(f'{cosmosis_dir}{data_dir}bin_{i}_{j}.txt')
			y_axis = ell*(ell+1)*c_ell / (2*np.pi)

			ax = plt.subplot2grid((7,7), (i,j-1))
			plt.plot(ell, y_axis, label=f'({i},{j})')
			plt.legend(frameon=False)
			plt.xscale('log')
			plt.yscale('log')

			if (i, j) == (1,1):
				plt.ylabel(r'$\ell*(\ell+1)*C_\ell / (2\pi)$')
			if (i,j) == (6,1):
				plt.xlabel(r'$\ell$')

plt.subplots_adjust(wspace = 0.4, hspace = 0.4)

plt.show()


