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

#exdir = '/gpfs/alpine/world-shared/phy122/sku/su412_f0_data/'
exdir = '/gpfs/alpine/proj-shared/csc143/jyc/summit/xgc-deeplearning/d3d_coarse_v2/'
with ad2.open(exdir + 'xgc.mesh.bp', 'r') as f:
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

#filename = (exdir + 'xgc.f0.10900.bp')
filename = (exdir + 'restart_dir/xgc.f0.00700.bp')
with ad2.open(filename, 'r') as f:
    i_f = f.read('i_f')
    i_f = np.moveaxis(i_f,1,2)
#    i_f = np.moveaxis(i_f, 0,1) # {N, phi, vx, vy}
    print (i_f.shape)

f_fsa = list()
in_fsa_idx = set([])

for i in range(len(psi_surf)):
    n = surf_len[i]
    k = surf_idx[i,:n]-1
    in_fsa_idx.update(k)
    f_fsa.append(i_f[:,k,:,:])
print(len(in_fsa_idx))
out_fsa_idx = list(set(range(nnodes)) - in_fsa_idx)
in_fsa_idx = np.fromiter(in_fsa_idx, dtype=np.int)
out_fsa_idx = np.fromiter(out_fsa_idx, dtype=np.int)
print("# of psi surface: ", len(psi_surf))
print("# nodes outside flux surface: ", len(out_fsa_idx))
print("# nodes inside flux surface: ", len(in_fsa_idx))
for i in range(len(out_fsa_idx)):
    k = out_fsa_idx[i]-1
    f_fsa.append(np.expand_dims(i_f[:,k,:,:], axis=1))
filename = 'd3d_coarse_v2_700_flx_phi.bp' 
with ad2.open(filename, "w") as fh:
    for i in range(len(psi_surf) + len(out_fsa_idx)):
        print(f_fsa[i].shape)
        fh.write("i_f", np.ndarray.flatten(f_fsa[i]), f_fsa[i].shape, [0,0,0,0], f_fsa[i].shape, end_step=True)

