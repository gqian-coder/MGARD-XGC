import numpy as np
import sys
#sys.path.append('/ccs/home/gongq/andes/indir/lib/python3.7/site-packages/adios2/')
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

def relative_node_abs_error(x, y):
    """
    relative point-wise L-inf error: max(|x_i - y_i|/|x_i|)
    """
    assert(x.shape == y.shape)
    absv = np.abs(x-y)
    maxv = np.max(np.abs(x), axis=(2,3))
    return (np.max(absv, axis=(2,3))/maxv)

def compute_diff(name, x, x_, res_folder):
    assert(x.shape == x_.shape)
    gb_L_inf  = relative_abs_error(x, x_)
    if (len(x.shape)==4):
        gb_L_inf = np.max(gb_L_inf, axis=(-1,-2))
        index    = np.argmax(gb_L_inf[0,:])
    else:
        index    = np.argmax(gb_L_inf[0,:])
    plt.figure()
    trimesh = tri.Triangulation(r, z, conn)
    plt.tricontourf(trimesh, np.mean(gb_L_inf, axis=0))
    plt.axis('equal');
    plt.axis('off')
    plt.colorbar()
    plt.savefig(res_folder+name+'_tri_rgb.png')
    plt.close()
    print("{}, shape = {}: gb L-inf error = {} at {}".format(res_folder+name, x.shape, np.max(gb_L_inf), index))
'''
    rel_L_inf = relative_node_abs_error(x,x_)
    plt.figure()
    trimesh = tri.Triangulation(r, z, conn)
    plt.tricontourf(trimesh, np.mean(x, axis=(0,2,3)))
    plt.axis('scaled')
    plt.colorbar()
    plt.savefig(res_folder+name+'_ori.png')
    plt.close()
    plt.figure()
    trimesh = tri.Triangulation(r, z, conn)
    plt.tricontourf(trimesh, np.mean(x_, axis=(0,2,3)))
    plt.axis('scaled')
    plt.colorbar()
    plt.savefig(res_folder+name+'_rct.png')
    plt.close()
    plt.figure()
    trimesh = tri.Triangulation(r, z, conn)
    plt.tricontourf(trimesh, np.mean(rel_L_inf, axis=0))
    plt.axis('equal');
    plt.axis('off')
    plt.colorbar()
    plt.savefig(res_folder+name+'_tri_rnd.png')
    plt.close()
'''

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
    f0_f = f.read('i_f')[0,:,:,:]
f0_f = np.expand_dims(f0_f, axis=0)
f0_f = np.moveaxis(f0_f,1,2)
nnodes = f0_f.shape[1]
n_surf = len(psi_surf) 
f0_f = f0_f[:,:nnodes,:,:]
#print (f0_f.shape)

rfile_path = '/gpfs/alpine/proj-shared/csc143/gongq/andes/MReduction/MGARD-XGC/build/'
fname = ['su412_10900_flx_phi.bp.fls.mgard.non.fls.']
eb = ['1e10', '5e10', '1e11', '5e11', '1e12', '5e12', '1e13', '5e13']
res_path = 'build/results/nonuniform_err_cr/'
#with ad2.open('/gpfs/alpine/proj-shared/csc143/gongq/andes/MReduction/MGARD-XGC/build/d3d_coarse_v2_700_flx_phi.bp.mgard', 'r') as f:
for ifile in range(len(fname)):
    for ieb in range(len(eb)):
        rct_file = rfile_path + fname[ifile] + eb[ieb] 
        print(rct_file)
        with ad2.open(rct_file, 'r') as f:
            f_rct = f.read('i_f')[0,:,:,:]
            f_rct = np.expand_dims(f_rct, axis=0)
#            print(f_rct.shape)
            in_fsa_idx = set([])
            _start = 0
            f0_g = np.zeros_like(f_rct)
            for i in range(n_surf):#len(psi_surf)):
                n = surf_len[i]
                k = surf_idx[i,:n]-1
                in_fsa_idx.update(k)
                f0_g[:,k,:,:] = f_rct[:,_start:_start+n,:,:]
                _start = _start + n
            del f_rct

#            relabserr = np.max(relative_abs_error(f0_f, f0_g), axis=(2,3))
#            print ('rel l-inf: ', relabserr.max())#, point_rel.max())
            res_folder = res_path + 'fls/' + eb[ieb] + '/'
#            if (ifile==0):
#                res_folder = res_path + 'gb/' + eb[ieb] + '/'
#            else:
#                res_folder = res_path + 'fls/' + eb[ieb] + '/'
            compute_diff("i-f", f0_f, f0_g, res_folder)

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
#            print (den_g.shape, u_para_g.shape, T_perp_g.shape, T_para_g.shape, n0_g.shape, T0_g.shape)
#        compute_diff("density_5d", den_f  , den_g   , res_folder)
#        compute_diff("u_para_5d", u_para_f, u_para_g, res_folder)
#        compute_diff("T_perp_5d", T_perp_f, T_perp_g, res_folder)
#        compute_diff("T_para_5d", T_para_f, T_para_g, res_folder)
        compute_diff("density_5d", np.sum(den_f   , axis=(-1,-2), dtype=np.float128), np.sum(den_g   , axis=(-1,-2), dtype=np.float128), res_folder)
        compute_diff("u_para_5d" , np.sum(u_para_f, axis=(-1,-2), dtype=np.float128), np.sum(u_para_g, axis=(-1,-2), dtype=np.float128), res_folder)
        compute_diff("T_perp_5d" , np.sum(T_perp_f, axis=(-1,-2), dtype=np.float128), np.sum(T_perp_g, axis=(-1,-2), dtype=np.float128), res_folder)
        compute_diff("T_para_5d" , np.sum(T_para_f, axis=(-1,-2), dtype=np.float128), np.sum(T_para_g, axis=(-1,-2), dtype=np.float128), res_folder)

