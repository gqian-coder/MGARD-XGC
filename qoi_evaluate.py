import numpy as np
import sys
sys.path.append('/ccs/home/gongq/indir/lib/python3.7/site-packages/adios2/')
import adios2 as ad2

import os
import subprocess

## module for xgc experiment: https://github.com/jychoi-hpc/xgc4py
import xgc4py
import f0_diag_test
xgcexp = xgc4py.XGC('/gpfs/alpine/proj-shared/csc143/jyc/summit/xgc-deeplearning/d3d_coarse_v2/')
xgc = f0_diag_test.XGC_f0_diag(xgcexp)

# read data
i_f = np.fromfile(, dtype=np.float64)
n_phi = xgcexp.nphi
n_nodes = xgcexp.mesh.nnodes
n_dv_perp = xgcexp.f0mesh.f0_nmu + 1
n_dv_para = 2 * xgcexp.f0mesh.f0_nvp + 1
assert(i_f.size == n_phi * n_nodes * n_dv_perp * n_dv_para)
# i_f = i_f.reshape([n_phi, n_nodes, n_dv_perp, n_dv_para])
i_f = i_f.reshape([n_nodes, n_phi, n_dv_perp, n_dv_para])
i_f = np.moveaxis(i_f, 0, 1)

density = np.zeros([n_phi, n_nodes])
u_para = np.zeros([n_phi, n_nodes])
T_perp = np.zeros([n_phi, n_nodes])
T_para = np.zeros([n_phi, n_nodes])
for i in range(n_phi):
	# no need to move axis as f is stored with velocity as the last two dimensions
	density[i], u_para[i], T_perp[i], T_para[i] = xgc.f0_diag(isp=1, f0_f=i_f[i])

n0 = density.copy()
T0 = (2.0*T_perp + T_para)/3.0
n0_avg, T0_avg = xgcexp.f0_avg_diag(0, n_nodes, n0, T0)

def compute_diff(name, x, x_):
	assert(x.shape == x_.shape)
	max_diff = np.max(np.abs(x - x_))
	max_v = np.max(np.abs(x))
	rmse = np.sqrt(np.mean((x - x_)**2))
	print("{}, shape = {}: L-inf error = {}, rmse = {}".format(name, x.shape, max_diff/max_v, rmse))

# compare
qoi = np.load("qoi.npz")
print("")
compute_diff("density", qoi['arr_0'], density)
compute_diff("u_para", qoi['arr_1'], u_para)
compute_diff("T_perp", qoi['arr_2'], T_perp)
compute_diff("T_para", qoi['arr_3'], T_para)
compute_diff("n0_avg", qoi['arr_4'], n0_avg)
compute_diff("T0_avg", qoi['arr_5'], T0_avg)
