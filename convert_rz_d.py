import os
import adios2 as ad2
import numpy as np
import matplotlib.pyplot as plt

exdir = '/Users/qg7/Documents/Projs/XGC Fusion/data files/low_res/su412/'

with ad2.open(exdir+'xgc.mesh.bp', 'r') as f:
	nnodes   = int(f.read('n_n', ))
	surf_idx = f.read('surf_idx')
	surf_len = f.read('surf_len')
	psi_surf = f.read('psi_surf')
	rz = f.read('rz')

r = rz[:,0]
z = rz[:,1]

dist = []
for i in range(len(psi_surf)):
	n    = surf_len[i]
	msk  = surf_idx[i,:n]-1
	msk1 = msk
	#cyclic shift of msk
	msk2 = np.roll(msk,-1) 
	d_tmp = (np.sqrt( (r[msk2] - r[msk1])**2 + (z[msk2] - z[msk1])**2 ))
	plt.plot(d_tmp)
	dist.append(d_tmp)

plt.show()

with ad2.open('rz_dist.bp', 'w') as fh:
	for i in range(len(psi_surf)):
		fh.write("dist", dist[i], [len(dist[i])], [0], [len(dist[i])], end_step=True)

