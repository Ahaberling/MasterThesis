# Importing modules
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

os.chdir('D:/Universitaet Mannheim/MMDS 7. Semester/Master Thesis/Outline/Data/Cleaning Robots/')

papers = pd.read_csv('cleaning_robot_EP_patents.csv', quotechar='"', skipinitialspace=True)
grid_results = pd.read_csv('lda_tuning_results.csv', quotechar='"', skipinitialspace=True)
# Print head

grid_results = grid_results.to_numpy()

#print(grid_results)
#print(max(grid_results[:,4]))

#print(grid_results[grid_results[:,4] == max(grid_results[:,4])])

#print(len(grid_results))

x = (int(len(grid_results)/2))
x2 = (int(len(grid_results)/2))

corpus75 = grid_results[0:x,:]
corpus100 = grid_results[x2:,:]

#print(corpus75)
print(corpus100)

# Alpha parameter   [0.01, 0.31, 0.61, 0.9099999999999999, 'symmetric', 'asymmetric']
# Beta parameter    [0.01, 0.31, 0.61, 0.9099999999999999, 'symmetric']

# Data for plotting

x = corpus100[:,1]
y001_1 = corpus100[:,1]


fig, ax = plt.subplots()
ax.plot(x, y)
ax.plot(x, y)
ax.plot(x, y2)

#ax.set(xlabel='time (s)', ylabel='voltage (mV)', title='About as simple as it gets, folks')
#ax.grid()

#fig.savefig("test.png")
#plt.show()



