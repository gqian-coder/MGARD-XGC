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
'''
plt.figure(figsize=[32,8])
for iphi in range(8):    
    plt.subplot(1, 8, iphi+1)
    trimesh = tri.Triangulation(r, z, conn)
    plt.tricontourf(trimesh, np.sum(f_new[iphi,:], axis=(1,2)))
    plt.axis('scaled')
    plt.axis('off')
    plt.ylim([0.0,0.3])
    plt.xlim([2.15,2.30])
    plt.title('iphi: %d'%iphi)
    plt.tight_layout()

#Group by flux surface index
#plt.figure(figsize=[8,16])

#trimesh = tri.Triangulation(r, z, conn)
#plt.triplot(trimesh, alpha=0.2)

#colormap = plt.cm.Dark2
f_fsa = list()
in_fsa_idx = set([]) 
for i in range(len(psi_surf)):
    n = surf_len[i]
    k = surf_idx[i,:n]-1
    in_fsa_idx.update(k)
    f_fsa.append(f_new[:,k,:,:])
#    plt.plot(r[k], z[k], '-', c=colormap(i%colormap.N))
#    plt.plot(r[k[0]], z[k[0]], 's', c=colormap(i%colormap.N))
#    plt.savefig('group_by_flux_surface.eps')

out_fsa_idx = list(set(range(nnodes)) - in_fsa_idx)
print("# nodes outside flux surface: ", len(out_fsa_idx))
print("# nodes inside flux surface: ", len(in_fsa_idx))
out_fsa = f_new[:,out_fsa_idx,:,:] 
with ad2.open("untwisted_flat_xgc.f0.00400.bp", "w") as fh:
    for i in range(len(psi_surf)):
        inSize = f_fsa[i].shape[0]*f_fsa[i].shape[1]*f_fsa[i].shape[2]*f_fsa[i].shape[3]
        print(i, np.count_nonzero(f_fsa[i])/39/39/8, f_fsa[i].shape)
        fh.write("i_f", f_fsa[i], [inSize], [0], [inSize], end_step=True)
    for i in range(len(out_fsa_idx)):
        fh.write("i_f", out_fsa[:,i,:,:], [8*39*39], [0], [8*39*39], end_step=True)
'''
