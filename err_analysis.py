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

with ad2.open('/gpfs/alpine/proj-shared/csc143/jyc/summit/xgc-deeplearning/d3d_coarse_v2/restart_dir/xgc.f0.00400.bp','r') as f:
    i_f_deep = f.read('i_f')
i_f_deep = np.moveaxis(i_f_deep,1,2)
print (i_f_deep.shape)

with ad2.open('/gpfs/alpine/proj-shared/csc143/gongq/andes/MReduction/MGARD-XGC/build/data/decompressed.bp', 'r') as f:
    i_f_whole = f.read('i_f')

with ad2.open('/gpfs/alpine/proj-shared/csc143/gongq/andes/MReduction/MGARD-XGC/build/decompressed_untwisted_4d.bp', 'r') as f:
    f_twisted_rct = f.read('i_f')

with ad2.open('/gpfs/alpine/proj-shared/csc143/gongq/andes/MReduction/MGARD-XGC/untwisted_4D.bp', 'r') as f:
    f_twisted_ori = f.read('i_f')

f_new = np.zeros_like(i_f_deep)
for iphi in range(8):
    od = nextnode_arr[iphi]
    f_new[iphi,:,:,:] = i_f_deep[iphi,od,:,:]

i_f_ori    = list()
in_fsa_idx = set([])
for i in range(len(psi_surf)):
    n = surf_len[i]
    k = surf_idx[i,:n]-1
    in_fsa_idx.update(k)
    i_f_ori.append(f_new[:,k,:,:])
#    print(i_f_ori[-1].shape)

out_fsa_idx = list(set(range(nnodes)) - in_fsa_idx)
out_fsa = f_new[:,out_fsa_idx,:,:]
for i in range(len(out_fsa_idx)):
    i_f_ori.append(np.expand_dims(out_fsa[:,i,:,:], axis=1))
#    print(i_f_ori[-1].shape)

max_nodes = 800
i_f_rct = list()
ioRead_rct = adios.DeclareIO("ioReader")
ibpStream = ioRead_rct.Open('/gpfs/alpine/proj-shared/csc143/gongq/andes/MReduction/MGARD-XGC/build/data/i_f.mgard.bp', ad2.Mode.Read)
max_nodes = max_nodes*8*39*39
for i in range(steps):
    ibpStream.BeginStep()
    var_i_f = ioRead_rct.InquireVariable("rct_i_f")
    i_f = np.zeros(max_nodes, dtype=np.double)
    ibpStream.Get(var_i_f, i_f, ad2.Mode.Sync)
    num_nodes = int(np.count_nonzero(i_f)/39/39/8)
    i_f_rct.append(np.moveaxis(np.reshape(i_f[:num_nodes*39*39*8], [num_nodes, 8, 39, 39]), [0,1,2,3], [1,0,2,3]))
    ibpStream.EndStep()
ibpStream.Close()


i_f_rct_nodes = list()
i_f_ori_nodes = list()
L_inf = list()
L_inf_perc  = list()
for i in range(steps):
#    print(i_f_ori[i].shape, i_f_rct[i].shape)
    for j in range(i_f_ori[i].shape[1]):
        i_f_ori_nodes.append(np.sum(i_f_ori[i][:,j,:,:]))
        i_f_rct_nodes.append(np.sum(i_f_rct[i][:,j,:,:]))
        L_inf.append(np.max(abs(i_f_rct[i][:,j,:,:] - i_f_ori[i][:,j,:,:])))
#        print(np.max(i_f_rct[i][:,j,:,:] - i_f_ori[i][:,j,:,:])/1e13)
        L_inf_perc.append(L_inf[-1]/np.max(i_f_ori[i][:,j,:,:])) 
        print(i, L_inf[-1]/1e13, L_inf_perc[-1]) 

i_f_whole       = np.moveaxis(i_f_whole, [0,1,2,3], [0,2,1,3])
i_f_whole_nodes =  np.sum(np.sum(np.sum(i_f_whole, axis=-1), axis=-1), axis=0)
i_f_deep_nodes  = np.sum(np.sum(np.sum(i_f_deep, axis=-1), axis=-1), axis=0)
print(i_f_deep.shape, i_f_deep_nodes.shape)
err_whole      = np.max(np.max(np.max(abs(i_f_whole - i_f_deep), axis=-1), axis=-1), axis=0)
err_whole_perc = err_whole/np.max(np.max(np.max(i_f_deep, axis=-1), axis=-1), axis=0) 

err_untwisted  = np.max(np.max(np.max(abs(f_twisted_rct - f_twisted_ori), axis=-1), axis=-1), axis=0)
err_untwisted_perc = err_untwisted/np.max(np.max(np.max(f_twisted_ori, axis=-1), axis=-1), axis=0)

plt.figure()
plt.plot(i_f_whole_nodes, label = 'original data' )
plt.plot(i_f_ori_nodes, label = 'twisted data')
plt.plot(i_f_rct_nodes, label = 'reconstruction by flux surface')
plt.legend()
plt.savefig('point_wise_err.eps')
plt.close()
plt.figure()
plt.plot(L_inf, label = 'by flux surface reconstruction')
plt.plot(err_whole, label = 'original reconstruction')
plt.plot(err_untwisted, label = 'twisted data reconstruction')
plt.legend()
print("max abs err -- original: {}, untwisted 4D: {},  by surface: {}".format(np.max(err_whole), np.max(err_untwisted), np.max(L_inf)))
plt.savefig('abs_err.eps')
plt.close()
plt.figure()
plt.plot(L_inf_perc, label = 'by flux surface reconstruction')
plt.plot(err_whole_perc, label = 'original reconstruction')
plt.plot(err_untwisted_perc, label = 'twisted data reconstruction')
plt.legend()
print("max perc err -- original: {}, untwisted 4D: {}, by surface: {}".format(np.max(err_whole_perc), np.max(err_untwisted_perc), np.max(L_inf_perc)))
plt.savefig('perc_err.eps')

