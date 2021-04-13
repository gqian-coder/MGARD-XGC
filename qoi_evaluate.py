import numpy as np
import sys
sys.path.append('/ccs/home/gongq/indir/lib/python3.7/site-packages/adios2/')
import adios2 as ad2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as tri

import os
import subprocess

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

def rmse_error(x, y):
    """
    root mean square error: square-root of sum of all (x_i-y_i)**2
    """
    assert(x.shape == y.shape)
    mse = np.mean((x-y)**2)
    return np.sqrt(mse)

def relative_rmse_error(x, y):
    """
    root mean square error: square-root of sum of all (x_i-y_i)**2
    """
    assert(x.shape == y.shape)
    mse = np.mean((x-y)**2)
    maxv = np.max(np.abs(x))
    return np.sqrt(mse)/maxv

def compute_diff(name, x, x_, datapath):
    assert(x.shape == x_.shape)
    plt.figure()
    trimesh = tri.Triangulation(r, z, conn)
#    plt.tricontourf(trimesh, np.mean(x, axis=(0,2,3)))
#    plt.axis('scaled')
#    plt.colorbar()
#    plt.savefig(name+'_ori.png')
#    plt.close()
#    plt.figure()
#    trimesh = tri.Triangulation(r, z, conn)
#    plt.tricontourf(trimesh, np.mean(x_, axis=(0,2,3)))
#    plt.axis('scaled')
#    plt.colorbar()
#    plt.savefig(name+'_rct.png')
#    plt.close()
    rel_err = relative_abs_error(x, x_)
    if (len(x.shape)==4):
        gb_L_inf = np.max(rel_err, axis=(-1,-2))
        index    = np.argmax(gb_L_inf[0,:])
        sort_err = np.argsort(gb_L_inf[0,:])
        print(sort_err[:10])
    else:
        gb_L_inf = rel_err
        index    = np.argmax(gb_L_inf[0,:]) 
        sort_err = np.argsort(gb_L_inf[0,:])
    plt.figure()
    trimesh = tri.Triangulation(r, z, conn)
    plt.tricontourf(trimesh, gb_L_inf[0,:])
    plt.axis('scaled')
    plt.colorbar()
    plt.savefig(datapath+'/'+name+'_tri_rgb.png')
    plt.close()
#    np.save(name+'.npy', rel_L_inf)
#    np.save(name+'_data.npy', x_)
#    np.save(name+'_rct.npy', x)
    print("{}, shape = {}: L-inf error = {} at {}".format(name, x.shape, np.max(gb_L_inf), index))

exdir = '/gpfs/alpine/world-shared/phy122/sku/su412_f0_data/' 
#exdir = '/gpfs/alpine/proj-shared/csc143/jyc/summit/xgc-deeplearning/d3d_coarse_v2/'
with ad2.open(exdir+'xgc.mesh.bp', 'r') as f:
# psi_surf: psi value of each surface
# surf_len: # of nodes of each surface
# surf_idx: list of node index of each surface
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

with ad2.open(exdir + 'xgc.f0.10900.bp', 'r') as f:
    f0_f = f.read('i_f')[0,:,:,:]
f0_f = np.expand_dims(f0_f, axis=0)
f0_f = np.moveaxis(f0_f, 1, 2)

## module for xgc experiment: https://github.com/jychoi-hpc/xgc4py
import xgc4py
xgcexp = xgc4py.XGC(exdir)
#xgcexp = xgc4py.XGC('/gpfs/alpine/world-shared/csc143/jyc/summit/d3d_coarse_small')

rct_path = 'build/'
eb = ['1e10', '5e10', '1e11', '5e11', '1e12', '5e12', '1e13', '5e13']
res_path = 'build/results/nonuniform_err_cr/4d/'
for i in range(8):
    filename = rct_path + "xgc.f0.10900.bp.mgard.4d." + eb[i]
    with ad2.open(filename, 'r') as f:
        print(filename)
        f0_g = f.read('i_f_4d')[0,:,:,:]
    res_folder = res_path+eb[i]
    idx_zr = np.where(f0_g < 0)
    f0_g[idx_zr] = 0
    f0_g = np.expand_dims(f0_g, axis=0)
    f0_g = np.moveaxis(f0_g, 1, 2)
    print(f0_f.shape, f0_g.shape)
    relabserr = np.max(relative_abs_error(f0_f, f0_g), axis=(2,3))
    #point_rel = np.max(relative_ptw_abs_error(f0_f, f0_g), axis=(2,3))
    print (relabserr.max(), relabserr.shape) #point_rel.max())

    n_phi = 1#f0_f.shape[0]
    n_vx, n_vy = f0_f.shape[-1], f0_f.shape[-2]
    f0_inode1 = 0
    ndata = f0_f.shape[-3]

    den_f    = np.zeros([n_phi, ndata, n_vx, n_vy])
    u_para_f = np.zeros([n_phi, ndata, n_vx, n_vy]) 
    T_perp_f = np.zeros([n_phi, ndata, n_vx, n_vy]) 
    T_para_f = np.zeros([n_phi, ndata, n_vx, n_vy]) 
    n0_f     = np.zeros([n_phi, ndata]) 
    T0_f     = np.zeros([n_phi, ndata])

    den_g    = np.zeros([n_phi, ndata, n_vx, n_vy]) 
    u_para_g = np.zeros([n_phi, ndata, n_vx, n_vy]) 
    T_perp_g = np.zeros([n_phi, ndata, n_vx, n_vy]) 
    T_para_g = np.zeros([n_phi, ndata, n_vx, n_vy]) 
    n0_g     = np.zeros([n_phi, ndata])
    T0_g     = np.zeros([n_phi, ndata])

    for iphi in range(n_phi):
        den_f[iphi,], u_para_f[iphi,], T_perp_f[iphi,], T_para_f[iphi,], n0_f[iphi,], T0_f[iphi,] =\
            xgcexp.f0_diag(f0_inode1=f0_inode1, ndata=ndata, isp=1, f0_f=f0_f[iphi,:])
        den_g[iphi,], u_para_g[iphi,], T_perp_g[iphi,], T_para_g[iphi,], n0_g[iphi,], T0_g[iphi,] =\
            xgcexp.f0_diag(f0_inode1=f0_inode1, ndata=ndata, isp=1, f0_f=f0_g[iphi,:])

    print (den_g.shape, u_para_g.shape, T_perp_g.shape, T_para_g.shape, n0_g.shape, T0_g.shape) 

    #n0_avg, T0_avg = xgcexp.f0_avg_diag(f0_inode1, ndata, n0, T0)
    #n0_avg_rct, T0_avg_rct = xgcexp.f0_avg_diag(f0_inode1, ndata, n0_rct, T0_rct)

    # compare
    compute_diff("i_f", f0_f, f0_g, res_folder) 
#    compute_diff("density_5d", den_f, den_g, rct_path+eb[i])
#    compute_diff("u_para_5d" , u_para_f, u_para_g, rct_path+eb[i])
#    compute_diff("T_perp_5d" , T_perp_f, T_perp_g, rct_path+eb[i])
#    compute_diff("T_para_5d" , T_para_f, T_para_g, rct_path+eb[i])
    compute_diff("density_5d", np.sum(den_f  , axis=(-1,-2)), np.sum(den_g   , axis=(-1,-2)), res_folder) 
    compute_diff("u_para_5d", np.sum(u_para_f, axis=(-1,-2)), np.sum(u_para_g, axis=(-1,-2)), res_folder) 
    compute_diff("T_perp_5d", np.sum(T_perp_f, axis=(-1,-2)), np.sum(T_perp_g, axis=(-1,-2)), res_folder) 
    compute_diff("T_para_5d", np.sum(T_para_f, axis=(-1,-2)), np.sum(T_para_g, axis=(-1,-2)), res_folder) 
    #compute_diff("n0_avg_5d", n0_avg_rct, n0_avg)
    #compute_diff("T0_avg_5d", T0_avg_rct, T0_avg)

