import numpy as np
import matplotlib.pyplot as plt
import adios2 as ad2
from math import atan, atan2, pi
import tqdm
import matplotlib.tri as tri

# psi_surf: psi value of each surface
# surf_len: # of nodes of each surface
# surf_idx: list of node index of each surface

with ad2.open('/Users/qg7/Documents/Projs/XGC Fusion/data files/low_res/d3d_coarse_v2/xgc.mesh.bp', 'r') as f:
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

i_T_para = np.load('/Users/qg7/Documents/Projs/XGC Fusion/data files/low_res/d3d_coarse_v2/T_para_400.npy')
print (i_T_para.shape)

T_new = np.zeros_like(i_T_para)
r_new = np.zeros_like(i_T_para)
z_new = np.zeros_like(i_T_para)
phi_  = np.zeros_like(i_T_para)
for iphi in range(8):
    od = nextnode_arr[iphi]
    T_new[:, iphi] = i_T_para[od, iphi]
    r_new[:, iphi] = r[od]
    z_new[:, iphi] = z[od]
    phi_[:, iphi ] = np.ones(nnodes)*(iphi+1)

T_r_z_phi_v = np.column_stack((r_new.flatten('F'), z_new.flatten('F'), phi_.flatten('F'), T_new.flatten('F')))
print(T_r_z_phi_v.shape)
np.savetxt('T_para.txt', T_r_z_phi_v, delimiter=',')
