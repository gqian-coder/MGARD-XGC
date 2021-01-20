import numpy as np
import sys
sys.path.append('/ccs/home/gongq/indir/lib/python3.7/site-packages/adios2/')
import matplotlib.pyplot as plt
import adios2 as ad2
from math import atan, atan2, pi
import tqdm
import matplotlib.tri as tri

# psi_surf: psi value of each surface
# surf_len: # of nodes of each surface
# surf_idx: list of node index of each surface

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
nextnode_arr.shape

with ad2.open('/gpfs/alpine/proj-shared/csc143/jyc/summit/xgc-deeplearning/d3d_coarse_v2/xgc.f3d.00400.bp','r') as f:
    i_T_para = f.read('i_T_para')
print (i_T_para.shape)
np.save('T_para_400.npy', i_T_para)

T_new = np.zeros_like(i_T_para)
for iphi in range(8):
    od = nextnode_arr[iphi]
    T_new[:, iphi] = i_T_para[od, iphi]
np.save('T_para_untwisted_400.npy', T_new)

with ad2.open('/gpfs/alpine/proj-shared/csc143/jyc/summit/xgc-deeplearning/d3d_coarse_v2/restart_dir/xgc.f0.00400.bp','r') as f:
    i_f = f.read('i_f')
np.save('i_f_400.npy', i_f)
