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
topics_res = pd.read_csv('lda_tuning_results_Mallet_default-Topics1000.csv', quotechar='"', skipinitialspace=True)
alpha_res = pd.read_csv('lda_tuning_results_Mallet_default-alpha.csv', quotechar='"', skipinitialspace=True)
opti_res = pd.read_csv('lda_tuning_results_Mallet_default-opti.csv', quotechar='"', skipinitialspace=True)
alphaopti_res = pd.read_csv('lda_tuning_results_Mallet_alphaOpti.csv', quotechar='"', skipinitialspace=True)

grid_results = grid_results.to_numpy()
topics_res = topics_res.to_numpy()
alpha_res = alpha_res.to_numpy()
opti_res = opti_res.to_numpy()
alphaopti_res = alphaopti_res.to_numpy()

os.chdir('D:/Universitaet Mannheim/MMDS 7. Semester/Master Thesis/Outline/Data/Cleaning Robots/GridSearch')
'''
gensim_topics = pd.read_csv('gensim_topics.csv', quotechar='"', skipinitialspace=True)
gensim_topics = gensim_topics.to_numpy()

mallet_topics = pd.read_csv('mallet_topics.csv', quotechar='"', skipinitialspace=True)
mallet_topics = mallet_topics.to_numpy()

gensim_330_alpha = pd.read_csv('gensim_330_alpha.csv', quotechar='"', skipinitialspace=True)
gensim_330_alpha = gensim_330_alpha.to_numpy()

mallet_330_alphaOpti = pd.read_csv('mallet_330_alphaOpti.csv', quotechar='"', skipinitialspace=True)
mallet_330_alphaOpti = mallet_330_alphaOpti.to_numpy()

gensim_topic_withAlpha01 = pd.read_csv('gensim_topic_withAlpha01.csv', quotechar='"', skipinitialspace=True)
gensim_topic_withAlpha01 = gensim_topic_withAlpha01.to_numpy()

mallet_topic_withAlpha01 = pd.read_csv('mallet_topic_withAlpha01.csv', quotechar='"', skipinitialspace=True)
mallet_topic_withAlpha01 = mallet_topic_withAlpha01.to_numpy()

gensim_topic_withAlpha025 = pd.read_csv('gensim_topic_withAlpha025.csv', quotechar='"', skipinitialspace=True)
gensim_topic_withAlpha025 = gensim_topic_withAlpha025.to_numpy()

gensim_topic_A01_extended = pd.read_csv('gensim_topic_A01_extended.csv', quotechar='"', skipinitialspace=True)
gensim_topic_A01_extended = gensim_topic_A01_extended.to_numpy()
'''
gensim_topics_A01 = pd.read_csv('gensim_topics_A01.csv', quotechar='"', skipinitialspace=True)
gensim_topics_A01 = gensim_topics_A01.to_numpy()

gensim_topics_A01_extended = pd.read_csv('gensim_topics_A01_extended.csv', quotechar='"', skipinitialspace=True)
gensim_topics_A01_extended = gensim_topics_A01_extended.to_numpy()

mallet_topics_A01 = pd.read_csv('mallet_topics_A01.csv', quotechar='"', skipinitialspace=True)
mallet_topics_A01 = mallet_topics_A01.to_numpy()

mallet_topics_A01_extended = pd.read_csv('mallet_topics_A01_extended.csv', quotechar='"', skipinitialspace=True)
mallet_topics_A01_extended = mallet_topics_A01_extended.to_numpy()

gensim_t330_alpha = pd.read_csv('gensim_t330_alpha.csv', quotechar='"', skipinitialspace=True)
gensim_t330_alpha = gensim_t330_alpha.to_numpy()

mallet_t330_op0_alpha = pd.read_csv('mallet_t330_op0_alpha.csv', quotechar='"', skipinitialspace=True)
mallet_t330_op0_alpha = mallet_t330_op0_alpha.to_numpy()

mallet_t330_a015_opti = pd.read_csv('mallet_t330_a015_opti.csv', quotechar='"', skipinitialspace=True)
mallet_t330_a015_opti = mallet_t330_a015_opti.to_numpy()

mallet_t330_a015_op1000_iter = pd.read_csv('mallet_t330_a015_op1000_iter.csv', quotechar='"', skipinitialspace=True)
mallet_t330_a015_op1000_iter = mallet_t330_a015_op1000_iter.to_numpy()




# OPTIMIZARTION INTER VALL WITH ALPHA 0.15 MALLET
x = mallet_t330_a015_op1000_iter[:,3]
y = mallet_t330_a015_op1000_iter[:,4]


fig, ax = plt.subplots()
#fig.subplots_adjust(bottom=0.2)
ax.set_ylim([0.35, 0.5])
#ax.set_yticklabels([1,4,5], fontsize=12)
ax.plot(x, y, color='darkblue')
#ax.set_xticklabels(x2)
#ax.scatter(x2, y2, c='green')
#ax.plot(x2_helper, y2, '.', color='darkblue')
#plt.xticks(rotation=45)
plt.xticks(np.arange(min(x), max(x)+1000, 1000))
#plt.xticks(np.arange(0, 2001, 400))
#plt.xticks(np.arange(min(x), max(x)+0.05, 0.05), labels=x2)
#plt.xlabel()
plt.xlabel("Number of training interations")
plt.ylabel("Coherency Score C_V")

#plt.show()
os.chdir('D:/Universitaet Mannheim/MMDS 7. Semester/Master Thesis/Outline/Plots')
plt.savefig('GrdiSearch_mallet_iter.png')
os.chdir('D:/Universitaet Mannheim/MMDS 7. Semester/Master Thesis/Outline/Data/Cleaning Robots/GridSearch')
plt.close()
plt.close()




'''
 THIS IS THE GENSIM ALPHA VISU
helper = [round(float(i), 2) for i in gensim_t330_alpha[:-3,1]]
#helper = round(helper, 2)
print(helper)

x = helper
y = gensim_t330_alpha[:-3,2]

x2 = gensim_t330_alpha[-3:,1]
x2_helper = [0.55, 0.6, 0.65]
y2 = gensim_t330_alpha[-3:,2]

xlabels = helper+['sym', 'asym', 'auto']
print(xlabels)

fig, ax = plt.subplots()
fig.subplots_adjust(bottom=0.2)
ax.set_ylim([0.25, 0.3])
#ax.set_yticklabels([1,4,5], fontsize=12)
ax.plot(x, y, color='darkblue')
#ax.scatter(x2, y2, c='green')
ax.plot(x2_helper, y2, '.', color='darkblue')
plt.xticks(rotation=45)
#plt.xticks(np.arange(min(x), max(x)+50, 50))
#plt.xticks(np.arange(0, 2001, 400))
plt.xticks(np.arange(min(x), max(x2_helper)+0.05, 0.05), labels=xlabels)
#plt.xlabel()
plt.xlabel("Alpha")
plt.ylabel("Coherency Score C_V")

#plt.show()
os.chdir('D:/Universitaet Mannheim/MMDS 7. Semester/Master Thesis/Outline/Plots')
plt.savefig('GrdiSearch_gensim_alpha.png')
os.chdir('D:/Universitaet Mannheim/MMDS 7. Semester/Master Thesis/Outline/Data/Cleaning Robots/GridSearch')
plt.close()

'''



'''
MALLET ALPHA SEARCH 

x = mallet_t330_op0_alpha[:,1]
y = mallet_t330_op0_alpha[:,3]

x2 = [round(float(i), 2) for i in list(np.arange(0.05, 0.51, 0.05))]
print(x2)

print(len(x))
print(len(x2))


fig, ax = plt.subplots()
#fig.subplots_adjust(bottom=0.2)
ax.set_ylim([0.35, 0.6])
#ax.set_yticklabels([1,4,5], fontsize=12)
ax.plot(x2, y, color='darkblue')
#ax.set_xticklabels(x2)
#ax.scatter(x2, y2, c='green')
#ax.plot(x2_helper, y2, '.', color='darkblue')
#plt.xticks(rotation=45)
plt.xticks(np.arange(min(x2), max(x2)+0.05, 0.05))
#plt.xticks(np.arange(0, 2001, 400))
#plt.xticks(np.arange(min(x), max(x)+0.05, 0.05), labels=x2)
#plt.xlabel()
plt.xlabel("Alpha")
plt.ylabel("Coherency Score C_V")

#plt.show()
os.chdir('D:/Universitaet Mannheim/MMDS 7. Semester/Master Thesis/Outline/Plots')
plt.savefig('GrdiSearch_mallet_alpha.png')
os.chdir('D:/Universitaet Mannheim/MMDS 7. Semester/Master Thesis/Outline/Data/Cleaning Robots/GridSearch')
plt.close()
plt.close()
'''

#plt.close()



#x = mallet_topics[:,0]
#y = mallet_topics[:,1]

fig, ax = plt.subplots()
ax.set_ylim([0.25, 0.3])
#ax.set_yticklabels([1,4,5], fontsize=12)
ax.plot(x, y)
#plt.show()

plt.close()

x = topics_res[:,1]
y = topics_res[:,2]

x_alpha = alpha_res[:,2]
y_alpha = alpha_res[:,3]

x_opti = opti_res[:,2]
y_opti = opti_res[:,3]

x_alphaopti_res1 = alphaopti_res[:,2]
x_alphaopti_res2 = alphaopti_res[:,3]
y_alphaopti_res = alphaopti_res[:,4]

print(x_opti)
print(y_opti)

#fig, ax = plt.subplots()
#ax.plot(x, y)
#ax.plot(x_alpha, y_alpha)
#ax.plot(x_opti, y_opti)
#ax.plot(300, 0.1)

plt.show()
plt.show()
plt.close()


ax = plt.axes(projection="3d")


z = alphaopti_res[:,4]
x = alphaopti_res[:,2]
y = alphaopti_res[:,3]

ax.plot3D(x,y,z)

plt.show()

'''
from matplotlib import cm
from matplotlib.ticker import LinearLocator

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})


# x = 1x40 -> 40x40
# y = 1x40 -> 40x40
# R = 40x40
# Z = 40x40

X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)

print(x)
print(Y)
print(Z)

surf = ax.plot_surface(X, Y, Z,
                       linewidth=0, antialiased=False)

ax.set_zlim(0, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')

fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
'''
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

