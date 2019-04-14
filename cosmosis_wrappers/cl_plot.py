import sys
import numpy as np
import matplotlib.pyplot as plt




def plot_cosmosis_cells(data_dir,bins):
	# ells
	ell = np.loadtxt(f'{data_dir}/ell.txt')
	
	fig = plt.figure(figsize=(10,8))
	for i in range(1,bins+1): # Cosmosis starts numbering at 1
		for j in range(1,bins+1):
			if i >= j: 
				c_ell = np.loadtxt(f'{data_dir}/bin_{i}_{j}.txt')
				y_axis = ell*(ell+1)*c_ell / (2*np.pi)

				ax = plt.subplot2grid((bins+1,bins+1), (i,j-1))
				
				# For the label
				plt.scatter(ell[0], y_axis[0], c='white',label=f'({i},{j})')
				
				plt.plot(ell, y_axis)
				plt.legend(frameon=False)
				plt.xscale('log')
				plt.yscale('log')

				if (i, j) == (1,1):
					plt.ylabel(r'$\ell*(\ell+1)*C_\ell / (2\pi)$')
				if (i,j) == (bins,1):
					plt.xlabel(r'$\ell$')

	plt.subplots_adjust(wspace = 0.4, hspace = 0.4)
	plt.tight_layout()
	plt.savefig('./TestFigures/somename.png')
	# plt.show()
	plt.close()



if __name__ == "__main__":

	data_dir = '/net/reusel/data1/osinga/master_research_project/saved_data/cosmosis/generate_cells_test/shear_cl'
	
	plot_cosmosis_cells(data_dir,bins=6)