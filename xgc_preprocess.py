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

for fstep in range(9):
    print("timestep: ", fstep)
    filename = ('/gpfs/alpine/proj-shared/csc143/jyc/summit/xgc-deeplearning/d3d_coarse_v2/restart_dir/xgc.f0.007' + str(fstep) + '0.bp')
    with ad2.open(filename, 'r') as f:
        i_f = f.read('i_f')
        i_f = np.moveaxis(i_f,1,2)
        print (i_f.shape)

    f_new = np.zeros_like(i_f)
    for iphi in range(8):
        od = nextnode_arr[iphi]
        f_new[iphi,:,:,:] = i_f[iphi,od,:,:]

    filename = ('/gpfs/alpine/proj-shared/csc143/gongq/XGC/d3d_coarse_v2/untwisted_4D_7'+ str(fstep) + '0.bp')
    with ad2.open(filename, "w") as fh:
        fh.write("i_f", f_new, f_new.shape,  [0,0,0,0],  f_new.shape)

    f_fsa = list()
    in_fsa_idx = set([])
    for i in range(len(psi_surf)):
        n = surf_len[i]
        k = surf_idx[i,:n]-1
    #    print(k)
        in_fsa_idx.update(k)
        f_fsa.append(f_new[:,k,:,:])

    out_fsa_idx = list(set(range(nnodes)) - in_fsa_idx)
    print("# nodes outside flux surface: ", len(out_fsa_idx))
    print("# nodes inside flux surface: ", len(in_fsa_idx))
    out_fsa = f_new[:,out_fsa_idx,:,:]
    filename = ('/gpfs/alpine/proj-shared/csc143/gongq/XGC/d3d_coarse_v2/untwisted_xgc.f0.007'+str(fstep)+'0.bp')
    with ad2.open(filename, "w") as fh:
        for i in range(len(psi_surf)):
            fh.write("i_f", f_fsa[i], f_fsa[i].shape, [0,0,0,0], f_fsa[i].shape, end_step=True)
        for i in range(len(out_fsa_idx)):
            fh.write("i_f", out_fsa[:,i,:,:], [8,1,39,39], [0,0,0,0,], [8,1,39,39], end_step=True)

