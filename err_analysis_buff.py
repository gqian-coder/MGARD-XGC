import numpy as np
import sys
sys.path.append('/ccs/home/gongq/indir/lib/python3.7/site-packages/adios2/')
import adios2 as ad2
import math 

from math import atan, atan2, pi
import tqdm
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

steps = 497 

adios = ad2.ADIOS()

with ad2.open('/gpfs/alpine/proj-shared/csc143/jyc/summit/xgc-deeplearning/d3d_coarse_v2/xgc.mesh.bp', 'r') as f:
#with ad2.open('/Users/qg7/Documents/Projs/XGC Fusion/data files/low_res/d3d_coarse_v2/xgc.mesh.bp', 'r') as f:
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

with ad2.open('/gpfs/alpine/proj-shared/csc143/gongq/andes/MReduction/MGARD-XGC/untwisted_4D_700.bp/', 'r') as f:
    f_twisted_ori = f.read('i_f')
print(f_twisted_ori.shape)

f_twisted_sf = list()
in_fsa_idx = set([])
for i in range(len(psi_surf)):
    n = surf_len[i]
    k = surf_idx[i,:n]-1
    in_fsa_idx.update(k)
    f_twisted_sf.append(f_twisted_ori[:,k,:,:])
#    print(f_twisted_sf[-1].shape)


out_fsa_idx = list(set(range(nnodes)) - in_fsa_idx)
print("# of psi surface: ", len(psi_surf))
print("# nodes outside flux surface: ", len(out_fsa_idx))
print("# nodes inside flux surface: ", len(in_fsa_idx))
for i in range(len(out_fsa_idx)):
    k = out_fsa_idx[i]
    f_twisted_sf.append(f_twisted_ori[:,k,:,:])

max_nodes = 800
f_twt_sf_rct = list()
ioRead_rct = adios.DeclareIO("ioReader")
ibpStream = ioRead_rct.Open('/gpfs/alpine/proj-shared/csc143/gongq/andes/MReduction/MGARD-XGC/build/new_i_f.mgard.bp', ad2.Mode.Read)
#ibpStream = ioRead_rct.Open('/Users/qg7/Documents/Projs/XGC Fusion/data files/low_res/d3d_coarse_v2/decompressed_twt_surface.bp', ad2.Mode.Read)
max_nodes = max_nodes*8*39*39
for i in range(steps):
    ibpStream.BeginStep()
    var_i_f = ioRead_rct.InquireVariable("rct_i_f")
    i_f = np.zeros(max_nodes, dtype=np.double)
    ibpStream.Get(var_i_f, i_f, ad2.Mode.Sync)
    num_nodes = int(np.count_nonzero(i_f)/39/39/8)
    print(i, num_nodes)
    f_twt_sf_rct.append(np.reshape(i_f[:num_nodes*39*39*8], [8, num_nodes, 39, 39]))
    ibpStream.EndStep()
ibpStream.Close()


f_twt_sf_rct_nodes = list()
L2_twt_sf	       = list() 
L_inf_twt_sf_perc  = list()
L_inf_twt_sf       = list()
f_data             = list()
'''
plt.figure()
plt.imshow(f_twisted_sf[1][4,1,:,:])
plt.savefig('read_twt.eps')
plt.close()
plt.figure()
plt.imshow(f_twt_sf_rct[1][4,5,:,:])
plt.savefig('read_twt_rct.eps')
plt.close()
'''

max_v = f_twisted_ori.max() 
i_f_rct = np.expand_dims(f_twt_sf_rct[0][:,4,:,:], axis=1)
print("max_v: ", max_v)
for i in range(steps):
#    print(f_twt_sf_rct[i].shape)
#    gb_max = np.max(f_twt_sf_rct[i])
    if (len(f_twisted_sf[i].shape)==4):
        n_elem = f_twisted_sf[i].shape[1]
    else:
        n_elem = 1
    for j in range(n_elem):
        if (len(f_twisted_sf[i].shape)==4):
            temp = f_twt_sf_rct[i][:,4+j,:,:] - f_twisted_sf[i][:,j,:,:]
            f_data.append(np.mean(f_twisted_sf[i][:,j,:,:]))
        else:
            temp = f_twt_sf_rct[i][:,4+j,:,:] - f_twisted_sf[i][:,:,:]
            f_data.append(np.mean(f_twisted_sf[i][:,:,:]))
        L_inf_twt_sf_perc.append(np.max(np.abs(temp/max_v)))
        L2_twt_sf.append(np.sqrt(np.mean(temp * temp)))
        L_inf_twt_sf.append(np.max(temp))
    if (i>0):
        i_f_rct = np.append(i_f_rct, f_twt_sf_rct[i][:,4:4+n_elem,:,:], axis=1)
#        print(i_f_rct.shape)
print(i_f_rct.shape)
np.save('i_f_twt_rct.npy', i_f_rct)
L2_twt_sf = np.array(L2_twt_sf)

plt.figure()
plt.plot(f_data, linewidth=1)
plt.savefig('i_f.eps')

plt.figure()
plt.plot(L2_twt_sf, linewidth=1, label = 'by flux surface reconstruction')
plt.legend()
plt.savefig('L2_err.eps')

plt.figure()
plt.plot(L_inf_twt_sf, linewidth=1, label = 'by flux surface reconstruction')
plt.legend()
plt.savefig('L_inf_err.eps')

plt.figure()
plt.plot(L_inf_twt_sf_perc, linewidth=1, label = 'by flux surface reconstruction')
plt.legend()
plt.savefig('L_inf_err_per.eps')
