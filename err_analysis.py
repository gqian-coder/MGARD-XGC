import numpy as np
import sys
sys.path.append('/ccs/home/gongq/indir/lib/python3.7/site-packages/adios2/')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import adios2 as ad2
import math 

from math import atan, atan2, pi
import tqdm
import matplotlib.tri as tri

def relative_abs_error(x, y):
    """
    relative L-inf error: max(|x_i - y_i|)/max(|x_i|)
    """
    assert(x.shape == y.shape)
    absv = np.abs(x-y)
    maxv = np.max(np.abs(x))
    return (absv/maxv)

def relative_ptw_abs_error(x, y):
    """
    relative point-wise L-inf error: max(|x_i - y_i|/|x_i|)
    """
    assert(x.shape == y.shape)
    absv = np.abs(x-y)
    return (absv/x)

steps = 101 
adios = ad2.ADIOS()

import xgc4py
exdir = '/gpfs/alpine/world-shared/phy122/sku/su412_f0_data/'
#exdir = '/gpfs/alpine/proj-shared/csc143/jyc/summit/xgc-deeplearning/d3d_coarse_v2/'
xgcexp = xgc4py.XGC(exdir)

with ad2.open(exdir+'xgc.mesh.bp', 'r') as f:
    nnodes = int(f.read('n_n', ))
    ncells = int(f.read('n_t', ))
    rz     = f.read('rz')
    conn   = f.read('nd_connect_list')
    psi    = f.read('psi')
    nextnode = f.read('nextnode')
    epsilon  = f.read('epsilon')
    node_vol = f.read('node_vol')
    node_vol_nearest = f.read('node_vol_nearest')
    psi_surf = f.read('psi_surf')
    surf_idx = f.read('surf_idx')
    surf_len = f.read('surf_len')

r = rz[:,0]
z = rz[:,1]

#with ad2.open(exdir + 'restart_dir/xgc.f0.00700.bp','r') as f:
with ad2.open(exdir+'xgc.f0.10900.bp', 'r') as f:
    f0_f = f.read('i_f')
f0_f = np.moveaxis(f0_f,1,2)
print (f0_f.shape)
nnodes = f0_f.shape[1]

#with ad2.open('/gpfs/alpine/proj-shared/csc143/gongq/andes/MReduction/MGARD-XGC/build/d3d_coarse_v2_700_flx_phi.bp.mgard', 'r') as f:
with ad2.open('/gpfs/alpine/proj-shared/csc143/gongq/andes/MReduction/MGARD-XGC/build/results/su412_10900_flx.bp.mgard', 'r') as f:
    f_rct = f.read('i_f')
f_rct = np.moveaxis(f_rct,0,1)
print(f_rct.shape)

in_fsa_idx = set([])
_start = 0
'''
f0_ff = np.zeros_like(f_rct)
f0_g = f_rct
for i in range(len(psi_surf)):
    n = surf_len[i]
    k = surf_idx[i,:n]-1
    in_fsa_idx.update(k)
    f0_ff[:,_start:_start+n,:,:] = f0_f[:,k,:,:]
    _start = _start + n

out_fsa_idx = list(set(range(nnodes)) - in_fsa_idx)
print(len(out_fsa_idx), _start)
for i in range(len(out_fsa_idx)):
    f0_ff[:,_start,:,:] = f0_f[:,out_fsa_idx[i]-1,:,:]
    _start = _start + 1
f0_f = f0_ff
'''
f0_g = np.zeros_like(f_rct)
for i in range(len(psi_surf)):
    n = surf_len[i]
    k = surf_idx[i,:n]-1
    in_fsa_idx.update(k)
    f0_g[:,k,:,:] = f_rct[:,_start:_start+n,:,:]
    _start = _start + n

out_fsa_idx = list(set(range(nnodes)) - in_fsa_idx)
print(len(out_fsa_idx), _start)
for i in range(len(out_fsa_idx)):
    f0_g[:,out_fsa_idx[i]-1,:,:] = f_rct[:,_start,:,:] 
    _start = _start + 1

del f_rct

relabserr = np.max(relative_abs_error(f0_f, f0_g), axis=(2,3))
idx = np.where(relabserr[0,:]>0.0001)
print(idx)
#f0_f[:,idx,:,:] = 1e-05
#f0_g[:,idx,:,:]= 1e-05
#relabserr = np.max(relative_abs_error(f0_f, f0_g), axis=(2,3))
#idx = np.where(relabserr[0,:]>0.0001)
#print(out_fsa_idx)
#point_rel = np.max(relative_ptw_abs_error(f0_f, f0_g), axis=(2,3))
print (relabserr.max())#, point_rel.max())

plt.figure()
trimesh = tri.Triangulation(r, z, conn)
#err_pt = np.zeros_like(relabserr[0,:])
#idx = np.where(relabserr[0,:]<0.0001)
#err_pt[idx] = relabserr[0,idx]
#plt.tricontourf(trimesh, err_pt)#, levels=20)
plt.tricontourf(trimesh, relabserr[0,:])#, levels=20)
plt.axis('equal');
plt.axis('off')
cbar = plt.colorbar()
plt.savefig('error_RZ.png')

n_phi = f0_f.shape[0]
f0_inode1 = 0
ndata = f0_f.shape[1]

den_f    = np.zeros_like(f0_f)
u_para_f = np.zeros_like(f0_f)
T_perp_f = np.zeros_like(f0_f)
T_para_f = np.zeros_like(f0_f)
n0_f     = np.zeros([n_phi, ndata])
T0_f     = np.zeros([n_phi, ndata])

den_g    = np.zeros_like(f0_f)
u_para_g = np.zeros_like(f0_f)
T_perp_g = np.zeros_like(f0_f)
T_para_g = np.zeros_like(f0_f)
n0_g     = np.zeros([n_phi, ndata])
T0_g     = np.zeros([n_phi, ndata])

for iphi in range(n_phi):
    den_f[iphi,], u_para_f[iphi,], T_perp_f[iphi,], T_para_f[iphi,], n0_f[iphi,], T0_f[iphi,] =\
        xgcexp.f0_diag(f0_inode1=f0_inode1, ndata=ndata, isp=1, f0_f=f0_f[iphi,:])
    den_g[iphi,], u_para_g[iphi,], T_perp_g[iphi,], T_para_g[iphi,], n0_g[iphi,], T0_g[iphi,] =\
         xgcexp.f0_diag(f0_inode1=f0_inode1, ndata=ndata, isp=1, f0_f=f0_g[iphi,:])

print (den_g.shape, u_para_g.shape, T_perp_g.shape, T_para_g.shape, n0_g.shape, T0_g.shape)

def compute_diff(name, x, x_):
        assert(x.shape == x_.shape)
        gb_L_inf  = relative_abs_error(np.sum(x, axis=(2,3)), np.sum(x_, axis=(2,3)))
#        rel_L_inf = relative_ptw_abs_error(np.sum(x,axis=(2,3)), np.sum(x_, axis=(2,3)))
#        plt.figure()
#        trimesh = tri.Triangulation(r, z, conn)
#        print(np.mean(rel_L_inf, axis=0).shape, np.mean(gb_L_inf, axis=0).shape)
#        plt.tricontourf(trimesh, np.mean(rel_L_inf, axis=0))
#        plt.axis('scaled')
#        plt.colorbar()
#        plt.savefig(name+'_tri_rpt.png')
#        plt.close()
        plt.figure()
        trimesh = tri.Triangulation(r, z, conn)
        plt.tricontourf(trimesh, np.mean(gb_L_inf, axis=0))
        plt.axis('equal');
        plt.axis('off')
        plt.colorbar()
        plt.savefig(name+'_tri_rgb.png')
        plt.close()
#        np.save(name+'.npy', rel_L_inf)
#        np.save(name+'_data.npy', x_)
#        np.save(name+'_rct.npy', x)
        print("{}, shape = {}: L-inf error = {}".format(name, x.shape, np.max(gb_L_inf)))

# compare
compute_diff("density_5d", den_f  , den_g)
compute_diff("u_para_5d", u_para_f, u_para_g)
compute_diff("T_perp_5d", T_perp_f, T_perp_g)
compute_diff("T_para_5d", T_para_f, T_para_g)
