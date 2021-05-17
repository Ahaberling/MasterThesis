# Importing modules
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

import sys
np.set_printoptions(threshold=sys.maxsize)

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
#corpus100 = grid_results[0:x,:]
corpus100 = grid_results[x2:,:]

#print(corpus75)
#print(corpus100)

# Alpha parameter   [0.01, 0.31, 0.61, 0.9099999999999999, 'symmetric', 'asymmetric']
# Beta parameter    [0.01, 0.31, 0.61, 0.9099999999999999, 'symmetric']


corpus100_001_001 = corpus100[(corpus100[:,2] == '0.01') & (corpus100[:,3] == '0.01') ,:]
corpus100_001_031 = corpus100[(corpus100[:,2] == '0.01') & (corpus100[:,3] == '0.31') ,:]
corpus100_001_061 = corpus100[(corpus100[:,2] == '0.01') & (corpus100[:,3] == '0.61') ,:]
corpus100_001_099 = corpus100[(corpus100[:,2] == '0.01') & (corpus100[:,3] == '0.9099999999999999') ,:]
corpus100_001_sym = corpus100[(corpus100[:,2] == '0.01') & (corpus100[:,3] == 'symmetric') ,:]

corpus100_031_001 = corpus100[(corpus100[:,2] == '0.31') & (corpus100[:,3] == '0.01') ,:]
corpus100_031_031 = corpus100[(corpus100[:,2] == '0.31') & (corpus100[:,3] == '0.31') ,:]
corpus100_031_061 = corpus100[(corpus100[:,2] == '0.31') & (corpus100[:,3] == '0.61') ,:]
corpus100_031_099 = corpus100[(corpus100[:,2] == '0.31') & (corpus100[:,3] == '0.9099999999999999') ,:]
corpus100_031_sym = corpus100[(corpus100[:,2] == '0.31') & (corpus100[:,3] == 'symmetric') ,:]

corpus100_061_001 = corpus100[(corpus100[:,2] == '0.61') & (corpus100[:,3] == '0.01') ,:]
corpus100_061_031 = corpus100[(corpus100[:,2] == '0.61') & (corpus100[:,3] == '0.31') ,:]
corpus100_061_061 = corpus100[(corpus100[:,2] == '0.61') & (corpus100[:,3] == '0.61') ,:]
corpus100_061_099 = corpus100[(corpus100[:,2] == '0.61') & (corpus100[:,3] == '0.9099999999999999') ,:]
corpus100_061_sym = corpus100[(corpus100[:,2] == '0.61') & (corpus100[:,3] == 'symmetric') ,:]

corpus100_099_001 = corpus100[(corpus100[:,2] == '0.9099999999999999') & (corpus100[:,3] == '0.01') ,:]
corpus100_099_031 = corpus100[(corpus100[:,2] == '0.9099999999999999') & (corpus100[:,3] == '0.31') ,:]
corpus100_099_061 = corpus100[(corpus100[:,2] == '0.9099999999999999') & (corpus100[:,3] == '0.61') ,:]
corpus100_099_099 = corpus100[(corpus100[:,2] == '0.9099999999999999') & (corpus100[:,3] == '0.9099999999999999') ,:]
corpus100_099_sym = corpus100[(corpus100[:,2] == '0.9099999999999999') & (corpus100[:,3] == 'symmetric') ,:]

corpus100_sym_001 = corpus100[(corpus100[:,2] == 'symmetric') & (corpus100[:,3] == '0.01') ,:]
corpus100_sym_031 = corpus100[(corpus100[:,2] == 'symmetric') & (corpus100[:,3] == '0.31') ,:]
corpus100_sym_061 = corpus100[(corpus100[:,2] == 'symmetric') & (corpus100[:,3] == '0.61') ,:]
corpus100_sym_099 = corpus100[(corpus100[:,2] == 'symmetric') & (corpus100[:,3] == '0.9099999999999999') ,:]
corpus100_sym_sym = corpus100[(corpus100[:,2] == 'symmetric') & (corpus100[:,3] == 'symmetric') ,:]

corpus100_asy_001 = corpus100[(corpus100[:,2] == 'asymmetric') & (corpus100[:,3] == '0.01') ,:]
corpus100_asy_031 = corpus100[(corpus100[:,2] == 'asymmetric') & (corpus100[:,3] == '0.31') ,:]
corpus100_asy_061 = corpus100[(corpus100[:,2] == 'asymmetric') & (corpus100[:,3] == '0.61') ,:]
corpus100_asy_099 = corpus100[(corpus100[:,2] == 'asymmetric') & (corpus100[:,3] == '0.9099999999999999') ,:]
corpus100_asy_sym = corpus100[(corpus100[:,2] == 'asymmetric') & (corpus100[:,3] == 'symmetric') ,:]



# Data for plotting

x = np.unique(corpus100[:,1])

y_001_001 = corpus100_001_001[:,4]
y_001_031 = corpus100_001_031[:,4]
y_001_061 = corpus100_001_061[:,4]
y_001_099 = corpus100_001_099[:,4]
y_001_sym = corpus100_001_sym[:,4]

y_031_001 = corpus100_031_001[:,4]
y_031_031 = corpus100_031_031[:,4]
y_031_061 = corpus100_031_061[:,4]
y_031_099 = corpus100_031_099[:,4]
y_031_sym = corpus100_031_sym[:,4]

y_061_001 = corpus100_061_001[:,4]
y_061_031 = corpus100_061_031[:,4]
y_061_061 = corpus100_061_061[:,4]
y_061_099 = corpus100_061_099[:,4]
y_061_sym = corpus100_061_sym[:,4]

y_099_001 = corpus100_099_001[:,4]
y_099_031 = corpus100_099_031[:,4]
y_099_061 = corpus100_099_061[:,4]
y_099_099 = corpus100_099_099[:,4]
y_099_sym = corpus100_099_sym[:,4]

y_sym_001 = corpus100_sym_001[:,4]
y_sym_031 = corpus100_sym_031[:,4]
y_sym_061 = corpus100_sym_061[:,4]
y_sym_099 = corpus100_sym_099[:,4]
y_sym_sym = corpus100_sym_sym[:,4]

y_asy_001 = corpus100_asy_001[:,4]
y_asy_031 = corpus100_asy_031[:,4]
y_asy_061 = corpus100_asy_061[:,4]
y_asy_099 = corpus100_asy_099[:,4]
y_asy_sym = corpus100_asy_sym[:,4]


fig, ax = plt.subplots()
ax.plot(x, y_001_001)
ax.plot(x, y_001_031)
ax.plot(x, y_001_061)
ax.plot(x, y_001_099)
ax.plot(x, y_001_sym)
plt.show()

fig, ax = plt.subplots()
ax.plot(x, y_031_001)
ax.plot(x, y_031_031)
ax.plot(x, y_031_061)
ax.plot(x, y_031_099)
ax.plot(x, y_031_sym)
plt.show()

fig, ax = plt.subplots()
ax.plot(x, y_061_001)
ax.plot(x, y_061_031)
ax.plot(x, y_061_061)
ax.plot(x, y_061_099)
ax.plot(x, y_061_sym)
plt.show()

fig, ax = plt.subplots()
ax.plot(x, y_099_001)
ax.plot(x, y_099_031)
ax.plot(x, y_099_061)
ax.plot(x, y_099_099)
ax.plot(x, y_099_sym)
plt.show()

fig, ax = plt.subplots()
ax.plot(x, y_sym_001)
ax.plot(x, y_sym_031)
ax.plot(x, y_sym_061)
ax.plot(x, y_sym_099)
ax.plot(x, y_sym_sym)
plt.show()

fig, ax = plt.subplots()
ax.plot(x, y_asy_001)
ax.plot(x, y_asy_031)
ax.plot(x, y_asy_061)
ax.plot(x, y_asy_099)
ax.plot(x, y_asy_sym)
plt.show()

# --------------------------

fig, ax = plt.subplots()
ax.plot(x, y_001_001)
ax.plot(x, y_001_031)
ax.plot(x, y_001_061)
ax.plot(x, y_001_099)
ax.plot(x, y_001_sym)

ax.plot(x, y_031_001)
ax.plot(x, y_031_031)
ax.plot(x, y_031_061)
ax.plot(x, y_031_099)
ax.plot(x, y_031_sym)

ax.plot(x, y_061_001)
ax.plot(x, y_061_031)
ax.plot(x, y_061_061)
ax.plot(x, y_061_099)
ax.plot(x, y_061_sym)

ax.plot(x, y_099_001)
ax.plot(x, y_099_031)
ax.plot(x, y_099_061)
ax.plot(x, y_099_099)
ax.plot(x, y_099_sym)

ax.plot(x, y_sym_001)
ax.plot(x, y_sym_031)
ax.plot(x, y_sym_061)
ax.plot(x, y_sym_099)
ax.plot(x, y_sym_sym)

ax.plot(x, y_asy_001)
ax.plot(x, y_asy_031)
ax.plot(x, y_asy_061)
ax.plot(x, y_asy_099)
ax.plot(x, y_asy_sym)
plt.show()


#ax.set(xlabel='time (s)', ylabel='voltage (mV)', title='About as simple as it gets, folks')
#fig.savefig("test.png")

