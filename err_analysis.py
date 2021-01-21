import numpy as np
import sys
sys.path.append('/ccs/home/gongq/indir/lib/python3.7/site-packages/adios2/')
import matplotlib.pyplot as plt
import adios2 as ad2
import math 

from math import atan, atan2, pi
import tqdm
import matplotlib.tri as tri

steps = 497

adios = ad2.ADIOS()

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

with ad2.open('/gpfs/alpine/proj-shared/csc143/jyc/summit/xgc-deeplearning/d3d_coarse_v2/restart_dir/xgc.f0.00400.bp','r') as f:
    f_ori = f.read('i_f')
#f_ori = np.load('/Users/qg7/Documents/Projs/XGC Fusion/data files/low_res/d3d_coarse_v2/i_f_400.npy')
f_ori = np.moveaxis(f_ori,1,2)
print (f_ori.shape)
nnodes = f_ori.shape[1]

with ad2.open('/gpfs/alpine/proj-shared/csc143/gongq/andes/MReduction/MGARD-XGC/build/data/decompressed.bp', 'r') as f:
#with ad2.open('/Users/qg7/Documents/Projs/XGC Fusion/data files/low_res/d3d_coarse_v2/decompressed.bp', 'r') as f:
    f_rct = f.read('i_f')
f_rct = np.moveaxis(np.reshape(f_rct, [8, 39,nnodes,39]), 1,2)

with ad2.open('/gpfs/alpine/proj-shared/csc143/gongq/andes/MReduction/MGARD-XGC/build/decompressed_twisted_4d.bp', 'r') as f:
#with ad2.open('/Users/qg7/Documents/Projs/XGC Fusion/data files/low_res/d3d_coarse_v2/decompressed_twisted_4d.bp', 'r') as f:
    f_twisted_rct = f.read('i_f')
f_twisted_rct = np.moveaxis(np.reshape(f_twisted_rct, [8, 39,nnodes,39]), 1,2)

with ad2.open('/gpfs/alpine/proj-shared/csc143/gongq/andes/MReduction/MGARD-XGC/twisted_4D.bp', 'r') as f:
    f_twisted_ori = f.read('i_f')
f_twisted_ori = np.moveaxis(np.reshape(f_twisted_ori, [8, 39,nnodes,39]), 1,2)

f_twisted_sf = list()
in_fsa_idx = set([])
for i in range(len(psi_surf)):
    n = surf_len[i]
    k = surf_idx[i,:n]-1
    in_fsa_idx.update(k)
    f_twisted_sf.append(f_twisted_ori[:,k,:,:])
    print(f_twisted_sf[-1].shape)

out_fsa_idx = list(set(range(nnodes)) - in_fsa_idx)
out_fsa = f_twisted_ori[:,out_fsa_idx,:,:]
for i in range(len(out_fsa_idx)):
    f_twisted_sf.append(np.expand_dims(out_fsa[:,i,:,:], axis=1))
#    print(f_twisted_sf[-1].shape)

max_nodes = 800
f_twisted_sf_rct = list()
ioRead_rct = adios.DeclareIO("ioReader")
ibpStream = ioRead_rct.Open('/gpfs/alpine/proj-shared/csc143/gongq/andes/MReduction/MGARD-XGC/build/data/i_f.mgard.bp', ad2.Mode.Read)
max_nodes = max_nodes*8*39*39
for i in range(steps):
    ibpStream.BeginStep()
    var_i_f = ioRead_rct.InquireVariable("rct_i_f")
    i_f = np.zeros(max_nodes, dtype=np.double)
    ibpStream.Get(var_i_f, i_f, ad2.Mode.Sync)
    num_nodes = int(np.count_nonzero(i_f)/39/39/8)
    f_twisted_sf_rct.append(np.moveaxis(np.reshape(i_f[:num_nodes*39*39*8], [num_nodes, 8, 39, 39]), [0,1,2,3], [1,0,2,3]))
    ibpStream.EndStep()
ibpStream.Close()


f_twisted_sf_rct_nodes = list()
L_inf = list()
L_inf_perc  = list()
for i in range(steps):
    for j in range(f_twisted_sf[i].shape[1]):
        f_twisted_sf_rct_nodes.append(np.sum(f_twisted_sf_rct[i][:,j,:,:]))
        L_inf.append(np.max(abs(f_twisted_sf_rct[i][:,j,:,:] - f_twisted_sf[i][:,j,:,:])))
        L_inf_perc.append(L_inf[-1]/np.max(f_twisted_sf[i][:,j,:,:])) 
        print(i, L_inf[-1]/1e13, L_inf_perc[-1]) 

f_twisted_ori_nodes = np.sum(np.sum(np.sum(f_twisted_ori, axis=-1), axis=-1), axis=0)
err_ori      = np.max(np.max(np.max(abs(f_ori - f_rct), axis=-1), axis=-1), axis=0)
err_ori_perc = err_ori/np.max(np.max(np.max(f_ori, axis=-1), axis=-1), axis=0) 

err_twisted  = np.max(np.max(np.max(abs(f_twisted_rct - f_twisted_ori), axis=-1), axis=-1), axis=0)
err_twisted_perc = err_twisted/np.max(np.max(np.max(f_twisted_ori, axis=-1), axis=-1), axis=0)

plt.figure()
plt.plot(f_twisted_ori_nodes, label = 'original twisted data')
plt.plot(f_twisted_sf_rct_nodes, alpha=0.7, label = 'reconstruction by flux surface')
plt.legend()
plt.savefig('point_wise_err.eps')
plt.close()
plt.figure()
plt.plot(L_inf, label = 'by flux surface reconstruction')
plt.plot(err_ori, alpha=0.7, label = 'original reconstruction')
plt.plot(err_twisted, alpha=0.7, label = 'twisted data reconstruction')
plt.legend()
print("max abs err -- original: {}, untwisted 4D: {},  by surface: {}".format(np.max(err_ori), np.max(err_twisted), np.max(L_inf)))
plt.savefig('abs_err.eps')
plt.close()
plt.figure()
plt.plot(L_inf_perc, label = 'by flux surface reconstruction')
plt.plot(err_ori_perc, alpha=0.7, label = 'original reconstruction')
plt.plot(err_twisted_perc, alpha=0.7, label = 'twisted data reconstruction')
plt.legend()
print("max perc err -- original: {}, untwisted 4D: {}, by surface: {}".format(np.max(err_ori_perc), np.max(err_twisted_perc), np.max(L_inf_perc)))
plt.savefig('perc_err.eps')

