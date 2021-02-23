import numpy as np
import matplotlib.pyplot as plt
#import adios2 as ad2
from math import atan, atan2, pi
import tqdm
import matplotlib.tri as tri

# psi_surf: psi value of each surface
# surf_len: # of nodes of each surface
# surf_idx: list of node index of each surface

qoi_file = "QoI.npz"
import xgc4py
import f0_diag_test
xgcexp = xgc4py.XGC('xgc_config/')
xgc = f0_diag_test.XGC_f0_diag(xgcexp)

'''
with ad2.open('/gpfs/alpine/proj-shared/csc143/jyc/summit/xgc-deeplearning/d3d_coarse_v2/xgc.mesh.bp', 'r') as f:
    nnodes = int(f.read('n_n', ))
    ncells = int(f.read('n_t', ))
    rz = f.read('rz')
    conn = f.read('nd_connect_list')
    psi = f.read('psi')
    nextnode = f.read('nextnode')
    epsilon = f.read('epsilon')
    node_vol = f.read('node_vol')
    node_vol_nearest = f.read('node_vol_nearest')
    psi_surf = f.read('psi_surf')
    surf_idx = f.read('surf_idx')
    surf_len = f.read('surf_len')

r = rz[:,0]
z = rz[:,1]
print (nnodes)

nextnode_list = list()
init = list(range(nnodes))
nextnode_list.append(init)
for iphi in range(1,8):
    prev = nextnode_list[iphi-1]
    current = [0,]*nnodes
    for i in range(nnodes):
        current[i] = nextnode[prev[i]]
        #print (i, prev[i], nextnode[prev[i]])
    nextnode_list.append(current)

nextnode_arr = np.array(nextnode_list)

filename = ('/gpfs/alpine/proj-shared/csc143/jyc/summit/xgc-deeplearning/d3d_coarse_v2/restart_dir/xgc.f0.00700.bp')
with ad2.open(filename, 'r') as f:
    i_f = f.read('i_f')
    i_f = np.moveaxis(i_f,1,2)
    print (i_f.shape)

f_new = np.zeros_like(i_f)
for iphi in range(8):
    od = nextnode_arr[iphi]
    f_new[iphi,:,:,:] = i_f[iphi,od,:,:]
'''
f_new = np.load('i_f_twt.npy')
density = np.zeros([n_phi, n_nodes])
u_para = np.zeros([n_phi, n_nodes])
T_perp = np.zeros([n_phi, n_nodes])
T_para = np.zeros([n_phi, n_nodes])
for i in range(n_phi):
    # no need to move axis as f is stored with velocity as the last two dimensions
    density[i], u_para[i], T_perp[i], T_para[i] = xgc.f0_diag(isp=1, f0_f=f_new[i])

n0 = density.copy()
T0 = (2.0*T_perp + T_para)/3.0
n0_avg, T0_avg = xgcexp.f0_avg_diag(0, n_nodes, n0, T0)
print("Store computed quantities...")
np.savez(qoi_file, density, u_para, T_perp, T_para, n0_avg, T0_avg)
