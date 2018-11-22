import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt


file = './modelsettings.csv'
csv_settings = pd.read_csv(file)

# Give all versions the same ID, so we can group by version
csv_settings['Version'] = np.asarray(csv_settings['Version'],dtype='int')

# Version, mean(detF train), mean(detF test) (over 3 runs)
mean_detF = csv_settings.groupby('Version')['Final detF train', 'Final detF test'].mean()

# The index of mean_detF is the version
plt.plot(mean_detF.index, mean_detF['Final detF train'], label='train')
plt.plot(mean_detF.index, mean_detF['Final detF test'], label='test')
plt.axhline(xmin = 0., xmax = 1., y = 50.
	, linestyle = 'dashed', color = 'black')
plt.ylabel('Det(F)')
plt.xlabel('Network version')
plt.ylim(0,100)
plt.legend(frameon=False)
plt.show()


fig, ax = plt.subplots(figsize = (15, 10))
plt.scatter(csv_settings['Version'], csv_settings['Final detF train'], label='train', alpha=0.5)
plt.scatter(csv_settings['Version'], csv_settings['Final detF test'], label='test', alpha=0.5)
plt.axhline(xmin = 0., xmax = 1., y = 50.
	, linestyle = 'dashed', color = 'black')
plt.ylabel('Det(F)',fontsize=18)
plt.xlabel('Network version',fontsize=18)
plt.ylim(0,100)
plt.legend(frameon=False,fontsize=14)
plt.xticks(csv_settings['Version'])
ax.tick_params(labelsize=12)
plt.savefig('./results_different_runs.png')
plt.show()